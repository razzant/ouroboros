"""move_to_project validates before emitting and never optimistically changes execution scope."""

from types import SimpleNamespace

import pytest

from ouroboros.tools.control import _move_to_project


def context(root, **kw):
    return SimpleNamespace(task_id="one", drive_root=root, budget_drive_root=root, task_metadata={},
                           task_contract={}, project_id="old", pending_events=[], event_queue=None, **kw)


def test_deferred_move_keeps_scope_until_durable_binding(tmp_path):
    ctx = context(tmp_path)
    result = _move_to_project(ctx, project_name="Room")
    assert result.startswith("SCOPE_UNCONFIRMED")
    assert ctx.project_id == "old"
    [event] = ctx.pending_events
    assert event["type"] == "ensure_project_scope" and event["task_id"] == "one"
    assert event["project_id"] == "room" and event["_project_admission"]["project"] is None


def test_subagent_and_empty_arguments_emit_nothing(tmp_path):
    ctx = context(tmp_path)
    assert "TOOL_ARG_ERROR" in _move_to_project(ctx, project_name="", project_id="")
    ctx.task_contract = {"lineage": {"delegation_role": "subagent"}}
    assert "subagents inherit" in _move_to_project(ctx, project_name="Room")
    assert not ctx.pending_events


@pytest.mark.parametrize("value", ["../escape", "Bad Name!"])
def test_invalid_id_is_refused(tmp_path, value):
    ctx = context(tmp_path)
    assert "TOOL_ARG_ERROR" in _move_to_project(ctx, project_id=value)
    assert not ctx.pending_events


def test_unreadable_binding_is_unknown_not_permission_to_move(tmp_path):
    """The binding store itself is damaged (torn, or a file that cannot be opened), whichever reader placement uses:
    refused, nothing emitted. The same move over a store that reads and names no binding for this work proceeds to
    its routing event."""
    from ouroboros.projects_registry import _bindings_path, create_project

    create_project(tmp_path, "room")  # a registry that reads: only the binding store differs between the moves
    ctx = context(tmp_path)
    store = _bindings_path(tmp_path)
    store.write_text("{torn", encoding="utf-8")
    assert "SCOPE_REJECTED" in _move_to_project(ctx, project_name="Room")
    store.unlink()
    store.mkdir()  # a directory where the store belongs: opening it fails with an OSError on every platform
    assert "SCOPE_REJECTED" in _move_to_project(ctx, project_name="Room")
    assert not ctx.pending_events and ctx.project_id == "old"
    store.rmdir()
    store.write_text('{"bindings": {}}', encoding="utf-8")
    assert _move_to_project(ctx, project_name="Room").startswith("SCOPE_UNCONFIRMED")
    assert [event["type"] for event in ctx.pending_events] == ["ensure_project_scope"] and ctx.project_id == "old"


def test_rejected_move_does_not_change_execution_scope(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr("ouroboros.tools.control._emit_and_wait_for_routing",
                        lambda *a: ("live", {"status": "rejected", "reason": "project_binding_failed"}))
    assert "SCOPE_REJECTED" in _move_to_project(ctx, project_name="Room")
    assert ctx.project_id == "old"
