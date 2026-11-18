"""One schedule_subagent call's independent shape problems come back in one refusal, before any route is selected;
the problems that depend on the selected row are named together after it (DEVELOPMENT 02: field, value and repair in
one reply, because a refusal that names one problem at a time is met again by the next)."""

from __future__ import annotations

import pytest

from ouroboros.tools import control, control_scheduling
from ouroboros.tools.registry import ToolContext
from tests._shared import configure_test_subagent, published_result as _published


def _ctx(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data")
    ctx.task_id, ctx.pending_events = "parent", []
    return ctx


def test_independent_shape_problems_come_back_in_one_refusal_before_any_route(tmp_path, monkeypatch):
    """Both directions: an unknown name, an invalid mode, a past deadline and a missing required field are named in
    one refusal and no route is selected; one problem alone reads exactly as it did; a valid call schedules."""
    selected = configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)
    with monkeypatch.context() as unrouted:
        unrouted.setattr(control_scheduling, "select_subagent_snapshot",
                         lambda *_a, **_k: pytest.fail("a call with shape problems selects no route"))
        refused = _published(ctx, "schedule_subagent", lambda: control._schedule_task(
            ctx, subagent_id=selected, objective="Examine the change", title="A title", mode="invented",
            deadline_at="2020-01-01T00:00:00Z"))
        alone = control._schedule_task(ctx, subagent_id=selected, objective="Examine the change")

    assert refused.code == "TOOL_ARG_ERROR" and refused.text == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): unsupported argument(s): title. Use the strict schema: subagent_id, "
        "objective, expected_output, optional role/context/constraints/memory_mode/workspace_root/effort and (for "
        "mutative children) write_surface/write_root/protected_paths_grant/external_tool_grants; mode must be ask, "
        "plan or agent; deadline_at is already in the past (2020-01-01T00:00:00Z); a child bound to it would finalize "
        "before doing any work; expected_output is required.")
    assert alone == "⚠️ TOOL_ARG_ERROR (schedule_subagent): expected_output is required."
    assert not ctx.pending_events
    control._schedule_task(ctx, subagent_id=selected, objective="Examine the change", expected_output="An explanation")
    assert [event["type"] for event in ctx.pending_events] == ["schedule_subagent"]


def test_problems_with_the_selected_row_are_named_together_after_it(tmp_path, monkeypatch):
    """Both directions: an API row that cannot take mode nor copy names both in one refusal; mode alone reads as it
    always did, under its own code; an agent_session row takes the mode and schedules."""
    selected = configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)
    base = {"subagent_id": selected, "objective": "Edit the notes", "expected_output": "The edited files"}

    both = _published(ctx, "schedule_subagent", lambda: control._schedule_task(
        ctx, **base, mode="agent", write_surface="external_workspace", directory_strategy="copy", scope_paths=["."]))
    alone = control._schedule_task(ctx, **base, mode="agent")

    assert both.code == "TOOL_ARG_ERROR" and both.meta == {"reason": "delegated_mode_unsupported"}
    assert both.text == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): delegated_mode_unsupported: mode requires an agent_session row; API "
        "children keep their ordinary task execution; directory_strategy=copy is unsupported for native/API children, "
        "which use shared files directly; select an agent_session actor for copy or workspace=in_place.")
    assert alone == ("⚠️ delegated_mode_unsupported: mode requires an agent_session row; API children keep their "
                     "ordinary task execution.")
    assert not ctx.pending_events
    session = configure_test_subagent(monkeypatch, subagent_id="session-scout", kind="agent_session",
                                      target="some-route=model")
    control._schedule_task(ctx, **{**base, "subagent_id": session}, mode="plan")
    assert [event["type"] for event in ctx.pending_events] == ["schedule_subagent"]
