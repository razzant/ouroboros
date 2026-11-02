"""Resumed direct Project work keeps the resource it was admitted with (#2547).

A direct turn has no queue preparation that pins a folder: its room lens is resolved
at each attempt. Once its basis survives a stop, a resumed attempt must not take a
folder the room acquired meanwhile, here through a sibling ``start_task`` promotion
that gives a folderless room its first folder by the registry's own compare-and-set.
The carried basis records the folder the turn was admitted with, ``""`` for none, and
every default consumer (tools, process cwd, the model's map and room rule, a child's
captured workspace) reads that. An explicit owner re-placement still re-selects.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_direct_project_turn_recovery import (
    PID,
    _publish_stop,
    _restore_next_generation,
    _resume_and_revalidate,
    _run_until_saved,
)
from tests.test_placement_pinned_resources import _assert_defaults, _started
from tests.test_swarm_host_admission import host as host

pytestmark = pytest.mark.serial
PROOF = "resource-proof.txt"


def _admit(host, folder=""):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import create_project
    from supervisor import worker_chat_lane, workers
    from tests._cancel_intents_shared import _CaptureQueue

    project = create_project(host.root, PID, name="Room 2547", working_dir=str(folder))
    text = "Continue this room work"
    origin = build_owner_message_ref(chat_id=project["chat_id"], client_message_id="resource-2547",
                                    ts="2026-11-01T00:00:00Z", text=text)
    workers.open_repo_writer_admission()
    actor = SimpleNamespace(handle_task=lambda task: [], _event_queue=_CaptureQueue())
    admitted = worker_chat_lane._admit_chat_task(actor, project["chat_id"], text, task_metadata={
        "project_id": PID, "client_message_id": "resource-2547", "origin_message_ref": origin,
        "origin_message_text": text})
    assert admitted is not None
    workers.REPO_DIR.mkdir(exist_ok=True)
    return project, admitted


def _write(host, task, monkeypatch, content):
    """The worker's own context builder, then a real write to the active workspace."""
    from supervisor import workers

    agent, ctx = _started(workers.REPO_DIR, host.root, task, monkeypatch)
    written = agent.tools.execute_result("write_file", {"root": "active_workspace", "path": PROOF, "content": content})
    assert written.status == "ok", written.text
    return agent, ctx, ctx.active_repo_dir()


def _stop_and_restore(host, admitted, monkeypatch):
    from supervisor import queue

    task = _run_until_saved(host.root, admitted)
    _publish_stop(host.root, admitted)
    assert _restore_next_generation(host.root, monkeypatch, queue) == []
    return task


def _sibling_assigns_the_first_folder(host, tmp_path, project):
    """An ordinary start_task promotion with an explicit folder: the room's first folder."""
    from ouroboros.projects_registry import get_project
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    folder = tmp_path / "new-room-folder"
    folder.mkdir()
    outcome = _handle_promote_chat_to_task({
        "task_id": "sibling-task", "routing_token": "sibling-task-token", "objective": "Work in this folder",
        "project_id": PID, "workspace_root": str(folder), "chat_id": project["chat_id"]}, host.ctx)
    assert outcome["status"] == "scheduled", outcome
    assert get_project(host.root, PID)["working_dir"] == str(folder)
    return folder


def _resumed(host, task_id):
    from supervisor import queue
    from supervisor.worker_assignment import _candidate_ineligible

    row = _resume_and_revalidate(queue, task_id)
    assert not row.get("_project_admission_restore_hold"), row.get("_project_admission_restore_hold")
    with queue._queue_lock:
        assert _candidate_ineligible(queue, row, 5.0, {}) is False
    return row


def test_resumed_work_keeps_its_folderless_resource_after_a_sibling_assigns_the_first_folder(
        host, tmp_path, monkeypatch):
    project, admitted = _admit(host)
    _agent, _ctx, before = _write(host, admitted["task"], monkeypatch, "before stop")
    assert before == host.root / "task_drives" / admitted["task"]["id"]
    task = _stop_and_restore(host, admitted, monkeypatch)
    folder = _sibling_assigns_the_first_folder(host, tmp_path, project)

    row = _resumed(host, task["id"])
    agent, ctx, after = _write(host, row, monkeypatch, "after resume")

    assert after == before, "resumed work kept the folderless resource it was admitted with"
    assert (before / PROOF).read_text() == "after resume" and not (folder / PROOF).exists()
    _assert_defaults(agent, ctx, row, before, "task_drive", monkeypatch)


def test_resumed_work_keeps_its_room_folder(host, tmp_path, monkeypatch):
    room = tmp_path / "room-folder"
    room.mkdir()
    _project, admitted = _admit(host, room)
    _agent, _ctx, before = _write(host, admitted["task"], monkeypatch, "before stop")
    assert before == room
    task = _stop_and_restore(host, admitted, monkeypatch)

    row = _resumed(host, task["id"])
    agent, ctx, after = _write(host, row, monkeypatch, "after resume")

    assert after == room and (room / PROOF).read_text() == "after resume"
    from ouroboros.context import build_runtime_section
    from ouroboros.tool_access import filesystem_affordance_map
    from ouroboros.tool_access_reads import capture_parent_workspace

    assert str(room) in agent.tools.execute_result("run_command", {"cmd": ["pwd"]}).text
    assert filesystem_affordance_map(ctx)["root_paths"]["active_workspace"] == str(room)
    assert capture_parent_workspace(ctx) == {"root": str(room), "mode": "", "source": "project_room"}
    monkeypatch.setattr("ouroboros.config.DATA_DIR", ctx.drive_root)
    assert "active_workspace is working_dir" in build_runtime_section(agent.env, row)


def test_the_owners_re_placement_selects_the_rooms_current_folder(host, tmp_path, monkeypatch):
    """Re-placing the work into its own room is a present-day decision: it takes the folder
    the room has now, as placement always did; only unplaced resumed work keeps its own."""
    from ouroboros.project_placement import move_task_to_project, placement_basis
    from supervisor import queue, workers
    from supervisor.task_admission import revalidate_project_holds

    project, admitted = _admit(host)
    task = _stop_and_restore(host, admitted, monkeypatch)
    folder = _sibling_assigns_the_first_folder(host, tmp_path, project)
    row = next(item for item in queue.PENDING if item["id"] == task["id"])
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda event: None))
    move_task_to_project(host.root, row["id"], PID, origin={"absent": "mid_task_no_origin"},
                         basis=placement_basis(host.root, row["id"], PID),
                         running=workers.RUNNING, pending=workers.PENDING, initiated_by="owner")
    with queue._queue_lock:
        revalidate_project_holds()

    row = _resumed(host, row["id"])
    _agent, _ctx, after = _write(host, row, monkeypatch, "after re-placement")

    assert row["_project_admission"]["project"]["working_dir"] == str(folder)
    assert after == folder and (folder / PROOF).read_text() == "after re-placement"
