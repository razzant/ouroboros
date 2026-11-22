"""One schedule_subagent call's independent shape problems come back in one refusal, before any route is selected;
the problems that depend on the selected row are named together after it (DEVELOPMENT 02: field, value and repair in
one reply, because a refusal that names one problem at a time is met again by the next)."""

from __future__ import annotations

import pytest

from ouroboros.tools import control, control_scheduling
from ouroboros.tools.registry import ToolContext
from tests._published_results import published_result as _published
from tests._shared import configure_test_subagent


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
            ctx, subagent_id=selected, objective="Examine the change", max_seconds=60, mode="invented",
            deadline_at="2020-01-01T00:00:00Z"))
        alone = control._schedule_task(ctx, subagent_id=selected, objective="Examine the change")

    assert refused.code == "TOOL_ARG_ERROR" and refused.text == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): unsupported argument(s): max_seconds. Use the strict schema: "
        "subagent_id, objective, expected_output, optional role/context/constraints/memory_mode/workspace_root/effort "
        "and (for mutative children) write_surface/write_root/protected_paths_grant/external_tool_grants; mode must be "
        "ask, plan or agent; deadline_at is already in the past (2020-01-01T00:00:00Z); a child bound to it would "
        "finalize before doing any work; expected_output is required.")
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


def test_independent_folder_field_problems_join_the_shape_refusal(tmp_path, monkeypatch):
    """Both directions: a scope path, a directory strategy and a workspace are each checked alone, so their problems
    come back in one refusal with the call's other shape problems; each alone reads exactly as it did; the copy rule,
    which reads two fields, still follows once they pass."""
    selected = configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)
    base = {"subagent_id": selected, "objective": "Read the notes", "expected_output": "Findings"}

    both = control._schedule_task(ctx, **base, scope_paths=[""], deadline_at="yesterday")
    assert both == ("⚠️ TOOL_ARG_ERROR (schedule_subagent): deadline_at must be an ISO-8601 UTC instant such as "
                    "2026-08-02T18:30:00Z (got: 'yesterday'); scope_paths must be an array of nonempty relative paths.")
    folder = control._schedule_task(ctx, **base, directory_strategy="old", scope_paths="../notes", workspace="other")
    assert folder == ("⚠️ TOOL_ARG_ERROR (schedule_subagent): directory_strategy must be direct or copy; scope_paths must "
                      "be an array of nonempty relative paths; workspace must be copy or in_place.")
    assert control._schedule_task(ctx, **base, scope_paths=[""]) == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): scope_paths must be an array of nonempty relative paths.")
    assert control._schedule_task(ctx, **base, directory_strategy="old") == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): directory_strategy must be direct or copy.")
    assert control._schedule_task(ctx, **base, workspace="other") == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): workspace must be copy or in_place.")
    assert control._schedule_task(ctx, **base, directory_strategy="copy") == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): copy requires scope_paths; use ['.'] to select the whole folder.")
    # workspace='in_place' names what is in the way and the one change that removes the conflict, choosing no form.
    lead = ("⚠️ TOOL_ARG_ERROR (schedule_subagent): workspace='in_place' needs an acting Git write_surface and no "
            "directory options; in the way: ")
    tail = ". Omitting workspace removes this conflict."
    assert control._schedule_task(ctx, **base, workspace="in_place", write_surface="external_workspace",
                                  scope_paths=["src"]) == lead + "scope_paths" + tail
    assert control._schedule_task(ctx, **base, workspace="in_place") == lead + "no write_surface" + tail
    assert not ctx.pending_events
    # The other direction: the documented defaults are not in the way, and dropping workspace schedules.
    acting = control._schedule_task(ctx, **base, workspace="in_place", write_surface="self_worktree",
                                    directory_strategy="direct", scope_paths=[])
    assert "in the way" not in acting
    control._schedule_task(ctx, **base)
    assert [event["type"] for event in ctx.pending_events][-1:] == ["schedule_subagent"]


def test_independent_file_and_origin_problems_join_the_deadline_refusal(tmp_path, monkeypatch):
    selected = configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)
    base = {"subagent_id": selected, "objective": "Read the notes", "expected_output": "Findings"}
    file_args = {"input_files": [{"root": "task_drive", "path": ""}]}
    origin_args = {"allowed_origins": ["studio.example"]}
    prefix = "⚠️ TOOL_ARG_ERROR (schedule_subagent): "
    file_error = "input_files is a list of {root, path} objects with a nonempty path."
    origin_error = "allowed_origins must contain exact HTTP(S) origins without paths, credentials or wildcards."

    with monkeypatch.context() as unrouted:
        unrouted.setattr(control_scheduling, "select_subagent_snapshot",
                         lambda *_a, **_k: pytest.fail("a call with shape problems selects no route"))
        refused = _published(ctx, "schedule_subagent", lambda: control._schedule_task(
            ctx, **base, deadline_at="yesterday", **file_args, **origin_args))
        assert refused.code == "TOOL_ARG_ERROR"
        assert refused.text == (prefix + "deadline_at must be an ISO-8601 UTC instant such as "
                                "2026-08-02T18:30:00Z (got: 'yesterday'); "
                                f"{file_error.rstrip('.')}; {origin_error}")
        assert control._schedule_task(ctx, **base, **file_args) == prefix + file_error
        assert control._schedule_task(ctx, **base, **origin_args) == prefix + origin_error
        assert not ctx.pending_events

    suffix = "input_files_readonly_only: input_files prepares inputs for a read-only child; omit it or write_surface."
    path = {"root": "task_drive", "path": "copper-notes.txt"}
    assert control._schedule_task(ctx, **base, input_files=[path], write_surface="external_workspace") == prefix + suffix
    assert not ctx.pending_events
    control._schedule_task(ctx, **base, input_files=[], allowed_origins=["https://studio.example", "https://studio.example:443/"])
    [event] = ctx.pending_events
    assert event["type"] == "schedule_subagent"
    assert event["task_contract"]["resource_policy"]["allowed_origins"] == ["https://studio.example:443"]


def test_title_is_one_refusal_naming_the_call_that_works(tmp_path, monkeypatch):
    """`title` is outside the schema: start_task's card name. Here no field means it (role becomes part of
    the child's assignment), so it reaches the handler only to be refused once with the call that works,
    beside the call's other shape problems; nothing is queued, and the same call without title schedules."""
    from ouroboros.tools.registry import ToolRegistry

    selected = configure_test_subagent(monkeypatch)
    ctx = _ctx(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path / "data")
    registry.set_context(ctx)
    base = {"subagent_id": selected, "objective": "Examine the change", "expected_output": "Findings"}

    for extra in ({}, {"role": "reviewer"}):
        refused = registry.execute("schedule_subagent", {**base, **extra, "title": "Change review"})
        assert refused == (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): title='Change review' is no schedule_subagent field: objective "
            "names the work, and the optional role becomes part of the child's assignment; send the call without "
            "title.")
    untitled = registry.execute("schedule_subagent", {"subagent_id": selected, "title": "Change review"})
    assert untitled == (
        "⚠️ TOOL_ARG_ERROR (schedule_subagent): title='Change review' is no schedule_subagent field: objective "
        "names the work and is required, and the optional role becomes part of the child's assignment; send the "
        "call without title; expected_output is required.")
    assert not ctx.pending_events
    registry.execute("schedule_subagent", base)
    assert [(event["type"], event.get("role")) for event in ctx.pending_events] == [("schedule_subagent", "")]
