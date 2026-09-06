"""Only host allocation establishes a task birth fact for archive read floors."""
from __future__ import annotations

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from tests.test_headless_task_title import admission as admission  # shared API fixture

pytestmark = pytest.mark.serial
CREATED = "2026-01-01T00:00:01+00:00"


def assert_lifecycle_keeps_creation(data, task_id):
    from ouroboros.headless import copy_child_task_result

    for status in ("running", "completed"):
        result = write_task_result(data, task_id, status, result=status)
        assert result["created_at"] == CREATED
    child = data.parent / (task_id + "-child")
    write_task_result(child, task_id, "completed", result="child result")
    mirrored = copy_child_task_result(data, {"id": task_id, "child_drive_root": str(child)})
    assert mirrored["created_at"] == CREATED


@pytest.mark.parametrize("supplied_id", [False, True])
def test_api_only_stamps_its_own_new_uuid_before_preparation(admission, monkeypatch, supplied_id):
    from ouroboros.gateway import tasks

    client, data, captured, _broadcasts = admission
    clock_calls = []
    monkeypatch.setattr(tasks, "utc_now_iso", lambda: clock_calls.append("birth") or CREATED)
    prepare = tasks.prepare_task_drive
    def prepare_after_allocation(*args, **kwargs):
        assert clock_calls == ([] if supplied_id else ["birth"])
        return prepare(*args, **kwargs)
    monkeypatch.setattr(tasks, "prepare_task_drive", prepare_after_allocation)
    body = {"description": "Inspect a local task"}
    if supplied_id:
        body["task_id"] = "supplied-task"
    response = client.post("/api/tasks", json=body)
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    initial = load_task_result(data, task_id)
    assert initial["status"] == "scheduled"
    if supplied_id:
        assert "created_at" not in initial and "created_at" not in captured[0]
    else:
        assert initial["created_at"] == captured[0]["created_at"] == CREATED
        assert_lifecycle_keeps_creation(data, task_id)


def test_tool_subagent_birth_precedes_drive_preparation(tmp_path, monkeypatch):
    from ouroboros.tools import control_scheduling
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "3")
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    clock_calls = []
    monkeypatch.setattr(control_scheduling, "utc_now_iso", lambda: clock_calls.append("stamp") or CREATED)
    prepare = control_scheduling._prepare_child_drive
    def prepare_after_allocation(*args, **kwargs):
        assert clock_calls == ["stamp"]
        return prepare(*args, **kwargs)
    monkeypatch.setattr(control_scheduling, "_prepare_child_drive", prepare_after_allocation)
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="parent")
    result = control_scheduling._schedule_task(
        ctx, subagent_id=subagent_id, objective="Inspect files", expected_output="findings", memory_mode="empty",
    )
    assert ctx.pending_events, result
    task_id = ctx.pending_events[0]["task_id"]
    initial = load_task_result(data, task_id)
    assert initial["status"] == "requested" and initial["created_at"] == CREATED
    assert_lifecycle_keeps_creation(data, task_id)


@pytest.mark.parametrize("supplied_id", [False, True])
def test_supervisor_stamps_only_a_locally_allocated_id(tmp_path, monkeypatch, supplied_id):
    from supervisor import events, events_schedule_task
    from tests.test_nested_rights_depth import _fake_ctx, _schedule_event

    monkeypatch.setattr(events_schedule_task, "utc_now_iso", lambda: CREATED)
    monkeypatch.setattr(events_schedule_task, "_find_duplicate_task", lambda *args, **kwargs: None)
    event = _schedule_event("supplied-task" if supplied_id else "", "", depth=0, drive_root=tmp_path)
    event["delegation_role"] = "root"
    enqueued = []
    events._handle_schedule_task(event, _fake_ctx(tmp_path, enqueued))
    assert enqueued
    task_id = enqueued[0]["id"]
    initial = load_task_result(tmp_path, task_id)
    assert initial["status"] == "scheduled"
    if supplied_id:
        assert "created_at" not in initial
    else:
        assert initial["created_at"] == CREATED
        assert_lifecycle_keeps_creation(tmp_path, task_id)


def test_final_only_and_legacy_updates_do_not_invent_creation(tmp_path):
    result = write_task_result(tmp_path, "final-only", "failed", ts="2026-01-01T01:00:00Z")
    assert "created_at" not in result
    write_task_result(tmp_path, "legacy", "running", ts="2025-01-01T00:00:00Z")
    result = write_task_result(tmp_path, "legacy", "completed")
    assert "created_at" not in result
