"""New committed work reopens its project once; running scheduled work needs no room binding."""
import dataclasses

import pytest

from ouroboros import dialogue_store, memory_view, projects_registry, task_results
from tests.test_project_closure import CLOSED
from tests.test_schedule_occurrence import _project, _row, q as q


def snapshot(root):
    spec = dataclasses.replace(memory_view.ROLE_DEFAULTS["integrator"], room_id="1")
    result = memory_view.capture_memory_view(root, {"id": "main-turn", "chat_id": 1}, spec)
    assert result.active
    return result


def test_scheduled_work_reopens_and_keeps_an_empty_room_live_without_binding_or_words(q):
    project = _project(q)
    room = str(project["chat_id"])
    projects_registry.update_project(q.root, "proj", closed=CLOSED)
    _row(q, project_id="proj", chat_id=project["chat_id"], intent={"kind": "system_repo"})
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert "closed" not in projects_registry.get_project(q.root, "proj")
    assert not projects_registry.project_binding_for_task(q.root, task["id"])
    assert not snapshot(q.root).live_rooms  # admission is scheduled, not running yet

    # Replay the accepted scheduler receipt after the mind has closed the project again.
    projects_registry.update_project(q.root, "proj", closed=CLOSED)
    q.pending.clear()
    q.queue.check_scheduled_tasks()
    assert [row["id"] for row in q.pending] == [task["id"]]
    assert projects_registry.get_project(q.root, "proj")["closed"] == CLOSED
    task_results.write_task_result(q.root, task["id"], "running")
    live = next(entry for entry in snapshot(q.root).live_rooms if entry["room_id"] == room)
    assert live["facts"]["running_tasks"] == [task["id"]] and live["facts"]["rows"] == 0
    assert projects_registry.get_project(q.root, "proj")["closed"] == CLOSED

    task_results.write_task_result(q.root, task["id"], "completed", result="Finished")
    dialogue_store.append(q.root, "chat", {"chat_id": project["chat_id"], "direction": "system",
        "type": "task_summary", "task_id": task["id"], "text": "Completed", "ts": "2026-10-29T12:00:00Z"})
    done = snapshot(q.root)
    assert room not in {entry["room_id"] for entry in done.live_rooms}
    assert room in {entry["room_id"] for entry in done.other_rooms}
    assert "1 open rooms;" in memory_view.render_room(done)


@pytest.mark.parametrize("failure", ["admission", "receipt"])
def test_refused_scheduled_work_does_not_open_the_project(q, monkeypatch, failure):
    project = _project(q)
    projects_registry.update_project(q.root, "proj", closed=CLOSED)
    _row(q, project_id="proj", chat_id=project["chat_id"], intent={"kind": "system_repo"})
    if failure == "admission":
        monkeypatch.setattr(q.queue, "_consciousness_admission_block", lambda *a: ("test_refusal", "Not admitted"))
    else:
        monkeypatch.setattr(task_results, "patch_document", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    q.queue.check_scheduled_tasks()
    assert not q.pending
    assert projects_registry.get_project(q.root, "proj")["closed"] == CLOSED


@pytest.mark.parametrize("status", ["scheduled", "running"])
def test_only_the_first_committed_task_admission_reopens(tmp_path, status):
    projects_registry.create_project(tmp_path, "proj")
    projects_registry.update_project(tmp_path, "proj", closed=CLOSED)
    task_results.write_task_result(tmp_path, "new-task", "requested", project_id="proj")
    task_results.write_task_result(tmp_path, "new-task", status, _field_projector=lambda *_: None)
    assert projects_registry.get_project(tmp_path, "proj")["closed"] == CLOSED
    task_results.write_task_result(tmp_path, "new-task", status)
    assert "closed" not in projects_registry.get_project(tmp_path, "proj")
    projects_registry.update_project(tmp_path, "proj", closed=CLOSED)
    task_results.write_task_result(tmp_path, "new-task", status)
    task_results.write_task_result(tmp_path, "new-task", "completed")
    assert projects_registry.get_project(tmp_path, "proj")["closed"] == CLOSED
