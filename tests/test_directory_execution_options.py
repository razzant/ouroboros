"""Caller-selected ordinary-folder geometry survives native scheduling transport."""
from __future__ import annotations

import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.tools.control_subagent_spec import _validated_schedule_fields
from ouroboros.delegate_shared import delegate_result


@pytest.mark.parametrize("options", [
    {}, {"directory_strategy": "direct"},
    {"directory_strategy": "copy", "scope_paths": ["."]},
    {"directory_strategy": "copy", "scope_paths": ["documents", "images/cover.png"]},
])
def test_directory_options_have_one_public_parameter_surface(options):
    from ouroboros.tools.control_subagent_spec import schedule_subagent_properties, schedule_subagent_param_names

    properties = schedule_subagent_properties()
    assert properties["directory_strategy"]["enum"] == ["direct", "copy"]
    assert properties["scope_paths"]["type"] == "array"
    assert {"directory_strategy", "scope_paths"} <= schedule_subagent_param_names()
    assert "REAL external Git" not in properties["write_root"]["description"]
    fields, error = _validated_schedule_fields({
        "objective": "Edit", "expected_output": "Files",
        "write_surface": "external_workspace", **options})
    assert not error
    assert {key: fields[key] for key in ("directory_strategy", "scope_paths") if key in fields} == options


@pytest.mark.parametrize("surface", [{}, {"write_surface": "read_only"}])
@pytest.mark.parametrize("options", [
    {}, {"directory_strategy": "direct"}, {"scope_paths": []},
    {"directory_strategy": "direct", "scope_paths": []},
])
def test_a_read_only_child_may_name_the_documented_default(surface, options):
    """#882: `direct` with no footprint is what omitting both already means.

    Nothing is normalized away here — the attested value is stored exactly as the
    parent passed it, because the same field carries a write-capable child's real
    choice. It simply is not a contradiction, so it is not refused.
    """
    fields, error = _validated_schedule_fields({
        "objective": "Audit", "expected_output": "Findings", **surface, **options})
    assert not error
    assert {key: fields[key] for key in ("directory_strategy", "scope_paths") if key in fields} == options


@pytest.mark.parametrize("surface", [{}, {"write_surface": "read_only"}])
@pytest.mark.parametrize("options", [
    {"directory_strategy": "copy", "scope_paths": ["."]},
    {"directory_strategy": "direct", "scope_paths": ["out"]},
    {"scope_paths": ["out"]},
])
def test_a_read_only_child_asking_for_real_geometry_is_refused_at_schedule_time(surface, options):
    """The contradiction is caught where the parent can still fix it in one move.

    Accepted here, it used to ride the envelope into the child's bootstrap and die
    at the host's pre-start — after a worker, a queue row and a paid round — with a
    receipt that forbade the child any other substrate. The message names the repair
    the parent can actually apply.
    """
    fields, error = _validated_schedule_fields({
        "objective": "Audit", "expected_output": "Findings", **surface, **options})
    assert not fields
    assert "TOOL_ARG_ERROR (schedule_subagent)" in error
    assert "omit directory_strategy and scope_paths" in error


@pytest.mark.parametrize("options", [
    {"directory_strategy": "copy", "scope_paths": ["."]},
    {"directory_strategy": "direct", "scope_paths": ["out"]},
])
def test_a_write_capable_child_keeps_real_geometry(options):
    """Unchanged for every acting surface: geometry is exactly what they are for."""
    for surface in ("self_worktree", "external_workspace", "genesis"):
        fields, error = _validated_schedule_fields({
            "objective": "Edit", "expected_output": "Files",
            "write_surface": surface, **options})
        assert not error, surface
        assert {key: fields[key] for key in ("directory_strategy", "scope_paths")} == options


@pytest.mark.parametrize("options", [
    {"directory_strategy": "invalid"}, {"directory_strategy": "copy"},
    {"directory_strategy": "copy", "scope_paths": []},
    {"scope_paths": "documents"}, {"scope_paths": [""]}, {"scope_paths": [1]},
    {"scope_paths": ["/absolute"]}, {"scope_paths": ["C:\\absolute"]},
    {"scope_paths": ["../outside"]},
])
def test_directory_options_refuse_an_unusable_scope_shape(options):
    fields, error = _validated_schedule_fields({"objective": "Edit", "expected_output": "Files", **options})
    assert not fields and "TOOL_ARG_ERROR" in error


def _schedule(tmp_path, monkeypatch, *, kind, options, surface="external_workspace"):
    from ouroboros.tools.control_scheduling import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    actor = configure_test_subagent(
        monkeypatch, kind=kind,
        target="codex=gpt-5.6-sol" if kind == "agent_session" else "openai/gpt-5.6-sol",
    )
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "1")
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "3")
    repo, data, folder = (tmp_path / name for name in ("system", "data", "documents"))
    for directory in (repo, data, folder):
        directory.mkdir()
    (folder / "draft.txt").write_text("owner draft\n", encoding="utf-8")
    event_queue = queue.Queue()
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="parent",
                      workspace_root=folder, workspace_mode="external")
    ctx.event_queue = event_queue
    ctx.task_metadata = {"root_task_id": "parent", "budget_drive_root": str(data)}
    acting = {"write_surface": surface, "write_root": str(folder)} if surface else {}
    response = _schedule_task(
        ctx, subagent_id=actor, objective="Revise draft.txt", expected_output="Updated draft",
        memory_mode="empty", **acting, **options,
    )
    return ctx, event_queue, response


def test_native_copy_refusal_precedes_child_side_effects(tmp_path, monkeypatch):
    ctx, events, response = _schedule(
        tmp_path, monkeypatch, kind="api_model",
        options={"directory_strategy": "copy", "scope_paths": ["."]},
    )
    assert "TOOL_ARG_ERROR" in response and "directory_strategy=copy is unsupported" in response
    assert events.empty()
    assert not list((ctx.drive_root / "task_results").glob("*.json"))
    assert not (ctx.workspace_root / ".git").exists()


@pytest.mark.parametrize("options", [
    {"directory_strategy": "copy", "scope_paths": ["."]},
    {"scope_paths": ["draft.txt"]},
])
def test_read_only_geometry_refusal_precedes_child_side_effects(tmp_path, monkeypatch, options):
    """The whole point of moving the check here: nothing is created to clean up."""
    ctx, events, response = _schedule(
        tmp_path, monkeypatch, kind="agent_session", options=options, surface="")
    assert "TOOL_ARG_ERROR" in response and "omit directory_strategy and scope_paths" in response
    assert events.empty()
    assert not list((ctx.drive_root / "task_results").glob("*.json"))


@pytest.mark.parametrize("options", [{}, {"directory_strategy": "direct"}, {"scope_paths": []}])
def test_a_read_only_child_naming_the_default_still_schedules(tmp_path, monkeypatch, options):
    """The surviving positive path: a read-only auditor is queued, not refused."""
    ctx, events, response = _schedule(
        tmp_path, monkeypatch, kind="agent_session", options=options, surface="")
    assert "Subagent request queued" in response, response
    event = events.get_nowait()
    assert event["write_surface"] == ""
    assert {key: event[key] for key in ("directory_strategy", "scope_paths") if key in event} == options


def test_a_read_only_native_child_may_no_longer_name_a_capture_footprint(tmp_path, monkeypatch):
    """The one capability this change narrows, pinned so it cannot drift unnoticed.

    A read-only NATIVE/API child used to schedule and run with `scope_paths`: the
    keys were inert there, because a native child never calls `delegate_start`. The
    new schedule-time guard keys on the write surface, not on the route, so that
    spelling is now a typed argument error that names the repair. Nothing documented
    is lost — the schema already said native children declare process outputs on
    their file/process tools — and a write-capable native child keeps the footprint
    verbatim, which is the half that must not move.
    """
    for case in ("read-only", "acting"):
        (tmp_path / case).mkdir()
    _, read_only_events, refused = _schedule(
        tmp_path / "read-only", monkeypatch, kind="api_model",
        options={"scope_paths": ["out"]}, surface="")
    assert "TOOL_ARG_ERROR (schedule_subagent)" in refused
    assert "omit directory_strategy and scope_paths" in refused
    assert read_only_events.empty()
    _, acting_events, queued = _schedule(
        tmp_path / "acting", monkeypatch, kind="api_model",
        options={"scope_paths": ["out"]}, surface="external_workspace")
    assert "Subagent request queued" in queued, queued
    assert acting_events.get_nowait()["scope_paths"] == ["out"]


@pytest.mark.parametrize("kind,options", [
    ("api_model", {}), ("api_model", {"directory_strategy": "direct"}),
    ("agent_session", {}), ("agent_session", {"directory_strategy": "direct"}),
    ("agent_session", {"directory_strategy": "copy", "scope_paths": ["draft.txt"]}),
    ("agent_session", {"directory_strategy": "copy", "scope_paths": ["."]}),
])
def test_directory_options_survive_schedule_result_dispatch_and_bootstrap(
    tmp_path, monkeypatch, kind, options,
):
    from ouroboros import subagent_runtime
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from ouroboros.task_results import load_task_result
    from supervisor import events
    from tests.test_nested_rights_depth import _fake_ctx

    parent, event_queue, response = _schedule(tmp_path, monkeypatch, kind=kind, options=options)
    assert not event_queue.empty(), response
    event = event_queue.get_nowait()
    tid = event["task_id"]
    keys = ("directory_strategy", "scope_paths")
    selected = lambda row: {key: row[key] for key in keys if key in row}
    assert selected(event) == options
    assert selected(load_task_result(parent.drive_root, tid)) == options
    enqueued = []
    supervisor = _fake_ctx(parent.drive_root, enqueued)
    supervisor.REPO_DIR = parent.repo_dir
    events._handle_schedule_task(event, supervisor)
    assert len(enqueued) == 1
    task = enqueued[0]
    assert selected(task) == selected(task["metadata"]) == options
    result = load_task_result(parent.drive_root, tid)
    assert result["status"] == "scheduled" and selected(result) == options
    assert task["workspace_root"] == str(parent.workspace_root)
    assert task["task_constraint"]["write_root"] == str(parent.workspace_root)
    assert not (parent.workspace_root / ".git").exists()

    starts = []
    def exact_start(ctx, prompt, spec):
        starts.append((ctx, prompt, spec))
        return delegate_result({"status": "started", "run_id": "directory-run"})
    monkeypatch.setattr(subagent_runtime, "exact_start", exact_start)
    child = SimpleNamespace(
        task_id=tid, drive_root=Path(task["drive_root"]), budget_drive_root=str(parent.drive_root),
        task_metadata=task["metadata"], workspace_root=Path(task["workspace_root"]),
        workspace_mode=task["workspace_mode"],
    )
    wake = bootstrap_before_context(child, task, SimpleNamespace(blocked=False))
    if kind == "api_model":
        assert wake == "" and starts == []
    else:
        assert json.loads(wake)["status"] == "configured_session_started"
        assert len(starts) == 1 and selected(starts[0][2]) == options
        assert selected(child._configured_actor_bootstrap) == options
        assert starts[0][0].workspace_root == parent.workspace_root
        assert starts[0][1] == child._configured_actor_bootstrap["canonical_work_order"]
        if "scope_paths" in task:
            task["scope_paths"].append("later-change")
            assert child._configured_actor_bootstrap["scope_paths"] == options["scope_paths"]
