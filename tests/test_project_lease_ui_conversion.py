"""Guard (post-v6.37.0): a live task converted to a project via the UI
("Turn into project", api_project_from_task) must hold its new project's one-writer
lease. The lease reads RUNNING[tid].task['project_id'], NOT the durable bindings, so
the convert path must update RUNNING too (SSOT helper shared with the in-task
ensure_project_scope path). Without it a concurrent same-project task could be
assigned — two writers per project."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _redirect_queue_snapshot(tmp_path, monkeypatch):
    """Every handler test here calls the real ``api_project_from_task``, which now
    persists the queue snapshot after its in-memory lease mark. Redirect
    ``supervisor.queue``'s snapshot writer to a throwaway temp path so no test can write
    (or silently swallow a failed write to) the live ``state/queue_snapshot.json`` under
    the real data root. A test that asserts on the snapshot re-points it explicitly."""
    from supervisor import state as state_mod
    import supervisor.queue as queue

    snap = tmp_path / "_queue_isolation" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)


def _request(tmp_path, body):
    async def _json():
        return body

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
        json=_json,
    )


def test_mark_task_project_running_and_pending():
    from ouroboros.project_lease import (
        candidate_is_leasable,
        mark_task_project,
        running_project_ids,
    )

    running = {"t1": {"task": {"id": "t1", "project_id": ""}}}
    pending = [{"id": "t2", "project_id": ""}]  # PENDING holds bare task dicts
    assert running_project_ids(running.values()) == set()           # no lane yet

    # RUNNING task -> occupies the lane immediately
    assert mark_task_project(running, pending, "t1", "proj-x") is True
    assert running["t1"]["task"]["project_id"] == "proj-x"
    assert running_project_ids(running.values()) == {"proj-x"}
    assert candidate_is_leasable({"id": "z", "project_id": "proj-x"}, {"proj-x"}) is False

    # PENDING task -> its own dict is scoped, so when assigned it will carry the lane
    # (assign_tasks reads the candidate's project_id and copies it into RUNNING)
    assert mark_task_project(running, pending, "t2", "proj-y") is True
    assert pending[0]["project_id"] == "proj-y"

    # no-op safety: not-present task, blank pid
    assert mark_task_project(running, pending, "missing", "proj-z") is False
    assert mark_task_project(running, pending, "t1", "") is False
    assert mark_task_project(running, None, "t1", "proj-x") is True   # pending=None tolerated


def test_ui_conversion_marks_running_task_as_lease_holder(tmp_path, monkeypatch):
    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.project_lease import candidate_is_leasable, running_project_ids
    import supervisor.workers as workers

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")

    # a still-RUNNING main-chat task with no project scope yet
    monkeypatch.setitem(workers.RUNNING, "tlive", {"task": {"id": "tlive", "project_id": ""}})
    assert running_project_ids(workers.RUNNING.values()) == set()

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "tlive", "id": "task-tlive", "objective_hint": "build it"},
    )))
    pid = json.loads(resp.body.decode("utf-8"))["project"]["id"]

    # the live task now carries the project_id in RUNNING -> holds the lane
    assert workers.RUNNING["tlive"]["task"]["project_id"] == pid
    assert pid in running_project_ids(workers.RUNNING.values())
    # so a concurrent same-project task is held out of assignment (one writer)
    assert candidate_is_leasable({"id": "other", "project_id": pid}, running_project_ids(workers.RUNNING.values())) is False


def test_ui_conversion_scopes_a_pending_task(tmp_path, monkeypatch):
    """codex finding: a task converted while still PENDING (not yet RUNNING) must get
    its project_id on the PENDING dict too — assign_tasks reads the candidate's own
    project_id and copies it into RUNNING, so without this the queued task starts
    unscoped and never holds its lane."""
    from ouroboros.gateway.projects import api_project_from_task
    import supervisor.workers as workers

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")

    pending_task = {"id": "tq", "project_id": "", "type": "task"}
    monkeypatch.setattr(workers, "PENDING", [pending_task])

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "tq", "id": "task-tq", "objective_hint": "queued work"},
    )))
    pid = json.loads(resp.body.decode("utf-8"))["project"]["id"]

    # the PENDING task dict is now scoped -> it will occupy the lane once assigned
    assert pending_task["project_id"] == pid
    assert workers.PENDING[0]["project_id"] == pid


def test_ui_conversion_persists_pending_scope_across_restart(tmp_path, monkeypatch):
    """scope finding: marking a PENDING converted task only in memory is not enough.
    restore_pending_from_snapshot rebuilds PENDING from state/queue_snapshot.json on
    restart, and assignment reads task['project_id'] from THERE (never the durable
    bindings). So the convert path must persist the snapshot right after the in-memory
    mark — otherwise a restart in the window (the snapshot is only rewritten on the next
    queue event) restores the task UNSCOPED and it never holds its lane."""
    from supervisor import state as state_mod
    from ouroboros.gateway.projects import api_project_from_task
    import supervisor.queue as queue
    import supervisor.workers as workers

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    snap = tmp_path / "state" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)

    pending_task = {"id": "tq", "project_id": "", "type": "task", "chat_id": 9}
    pending_list = [pending_task]
    # Point BOTH the gateway's view (workers.*) and the snapshot writer (queue.*) at the
    # SAME live list, as init_queue_refs does in production.
    monkeypatch.setattr(workers, "PENDING", pending_list)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", pending_list)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(state_mod, "QUEUE_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "tq", "id": "task-tq", "objective_hint": "queued work"},
    )))
    pid = json.loads(resp.body.decode("utf-8"))["project"]["id"]

    # the mark reached the persisted snapshot, not just the in-memory list
    assert snap.exists()
    saved = json.loads(snap.read_text(encoding="utf-8"))
    assert saved["pending"][0]["task"]["project_id"] == pid

    # simulate a restart: empty live queue, restore from the snapshot -> STILL scoped
    pending_list.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert pending_list[0]["project_id"] == pid
