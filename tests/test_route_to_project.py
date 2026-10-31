"""The retired fork/picker tool cannot create new work; historical cards have their own tests."""

import inspect
from types import SimpleNamespace

import pytest

from ouroboros.tools.control import _start_task, get_tools
from ouroboros.tools.registry import ToolContext, ToolRegistry


def test_only_separate_start_and_same_work_move_are_public(tmp_path):
    entries = {entry.name: entry for entry in get_tools()}
    assert {"start_task", "move_to_project"} <= entries.keys()
    assert not {"promote_chat_to_task", "ensure_project_scope", "route_to_project"} & entries.keys()
    assert "predecessor_task_id" not in inspect.signature(_start_task).parameters
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    for retired in ("promote_chat_to_task", "ensure_project_scope", "route_to_project"):
        assert registry.get_schema_by_name(retired) is None
    for old, new in (("promote_chat_to_task", "start_task"), ("ensure_project_scope", "move_to_project")):
        registry.set_context(ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data",
                                        task_metadata={"disabled_tools": [old]}))
        assert registry.get_schema_by_name(new) is None
        assert "RESOURCE_CONSTRAINT_BLOCKED" in registry.execute(new, {})


def test_start_is_separate_and_retains_callers_obligations(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result

    monkeypatch.setattr("ouroboros.tools.control_events._PROMOTE_CONFIRM_TIMEOUT_SEC", 0)
    ctx = SimpleNamespace(task_id="caller", drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_metadata={"force_plan": True}, task_contract={}, project_id="",
                          pending_events=[], event_queue=None, unconfirmed_routing_acts=[])
    result = _start_task(ctx, "Separate experiment", title="Experiment", reasoning_effort="")
    assert result.startswith("Accepted, waiting to start:")
    [event] = ctx.pending_events
    assert event["task_id"] != "caller" and event["type"] == "promote_chat_to_task"
    assert not {"predecessor_task_id", "predecessor_authority_source", "force_plan_transferred_from"} & event.keys()
    assert ctx.task_metadata["force_plan"] is True
    assert load_task_result(tmp_path, event["task_id"])["promotion_intent"] == event
    with pytest.raises(TypeError, match="predecessor_task_id"):
        _start_task(ctx, "Cannot continue a settled result here", title="Old selector", predecessor_task_id="old")


def test_start_never_adopts_an_origin_siblings_project(tmp_path):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project, project_binding_for_task

    text = "Owner request"
    ref = build_owner_message_ref(chat_id=1, client_message_id="owner", ts="2026-10-16T12:00:00Z", text=text)
    create_project(tmp_path, "sibling-room")
    bind_task_to_project(tmp_path, "sibling", "sibling-room", origin={"ref": ref, "text": text})
    ctx = SimpleNamespace(task_id="caller", drive_root=tmp_path, budget_drive_root=tmp_path, project_id="",
                          task_metadata={"origin_message_ref": ref, "origin_message_text": text},
                          pending_events=[], event_queue=None, unconfirmed_routing_acts=[])
    _start_task(ctx, "Another experiment", title="Separate")
    [event] = ctx.pending_events
    assert event["project_id"] == "" and event["source_ref"] == ref
    assert project_binding_for_task(tmp_path, "caller") is None
    assert project_binding_for_task(tmp_path, "sibling")["project_id"] == "sibling-room"
