"""The subagent scheduling surface: schedule_subagent, wait_task, get_task_result.

Split verbatim out of ``tests/test_tool_capabilities.py`` by theme. This
module owns which control tools a scheduling principal sees and may call:
core membership, registry and schema visibility, the required-capability
fail-fast, the top-level control surface under workspace focus, executor-ref
inheritance, and the capability omission manifest.
"""
import json


# ---------------------------------------------------------------------------
# schedule_subagent core classification tests
# ---------------------------------------------------------------------------


def test_schedule_subagent_in_core():
    """schedule_subagent is core for first-class parallel delegation."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "schedule_subagent" in CORE_TOOL_NAMES


def test_wait_task_in_core():
    """wait_task/wait_tasks are core so delegated work can be joined."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "wait_task" in CORE_TOOL_NAMES
    assert "wait_tasks" in CORE_TOOL_NAMES


def test_get_task_result_in_core():
    """get_task_result is core so child handoffs can be read."""
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES
    assert "get_task_result" in CORE_TOOL_NAMES


def test_schedule_subagent_available_in_registry():
    """schedule_subagent must still be registered."""
    from ouroboros.tools.registry import ToolRegistry
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    all_names = {t["function"]["name"] for t in registry.schemas()}
    assert "schedule_subagent" in all_names, (
        "schedule_subagent must be discoverable via list_available_tools / enable_tools"
    )


def test_schedule_subagent_in_initial_schemas():
    """schedule_subagent appears in parent initial schemas as a core tool."""
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tool_policy import initial_tool_schemas
    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    registry = ToolRegistry(repo_dir=tmp, drive_root=tmp)
    names = {s["function"]["name"] for s in initial_tool_schemas(registry)}
    assert "schedule_subagent" in names
    assert {"peek_task", "cancel_task", "discard_child_result"} <= names
    schedule_schema = next(s for s in initial_tool_schemas(registry) if s["function"]["name"] == "schedule_subagent")
    props = schedule_schema["function"]["parameters"]["properties"]
    assert "required_capabilities" in props
    assert "shell" in props["required_capabilities"]["items"]["enum"]


def test_schedule_subagent_required_capabilities_fail_fast_for_readonly(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    ctx.repo_dir.mkdir(parents=True)
    ctx.drive_root.mkdir(parents=True)
    result = _schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Need git diff",
        expected_output="diff summary",
        required_capabilities=["shell", "vcs"],
        write_surface="read_only",
    )
    assert "SUBAGENT_CAPABILITY_MISMATCH" in result


def test_schedule_subagent_required_delegate_capability_is_satisfied_for_readonly(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    events = []
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    ctx.repo_dir.mkdir(parents=True)
    ctx.drive_root.mkdir(parents=True)
    ctx.pending_events = events
    ctx.task_id = "parent1"
    result = _schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Delegate deeper readonly work",
        expected_output="child id",
        required_capabilities=["delegate"],
        write_surface="read_only",
    )
    assert "SUBAGENT_CAPABILITY_MISMATCH" not in result
    assert events and events[0]["type"] == "schedule_subagent"
    assert events[0]["required_capabilities"] == ["delegate"]


def test_schedule_subagent_required_vcs_capability_is_satisfied_for_readonly(tmp_path, monkeypatch):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)

    events = []
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    ctx.repo_dir.mkdir(parents=True)
    ctx.drive_root.mkdir(parents=True)
    ctx.pending_events = events
    ctx.task_id = "parent1"
    result = _schedule_task(
        ctx,
        subagent_id=subagent_id,
        objective="Inspect git status in readonly child",
        expected_output="status summary",
        required_capabilities=["vcs"],
        write_surface="read_only",
    )
    assert "SUBAGENT_CAPABILITY_MISMATCH" not in result
    assert events and events[0]["required_capabilities"] == ["vcs"]


def test_local_readonly_subagent_initial_schemas_are_allowlisted(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    from ouroboros.tool_policy import initial_tool_schemas, list_non_core_tools
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(
        ToolContext(
            repo_dir=tmp_path,
            drive_root=tmp_path,
            task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False),
        )
    )

    names = {s["function"]["name"] for s in initial_tool_schemas(registry)}
    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES <= names
    assert "enable_tools" not in names
    assert "schedule_subagent" in names
    assert "verify_and_record" not in names
    assert "write_file" not in names
    assert "run_command" not in names
    assert "browse_page" in names
    assert "browser_action" in names
    schemas = {s["function"]["name"]: s["function"] for s in initial_tool_schemas(registry)}
    for tool_name in ("read_file", "list_files", "search_code"):
        root_enum = schemas[tool_name]["parameters"]["properties"]["root"]["enum"]
        assert "user_files" not in root_enum
    assert set(schemas["search_code"]["parameters"]["properties"]["root"]["enum"]) == {"active_workspace", "system_repo", "skill_payload"}
    action_schema = schemas["browser_action"]["parameters"]["properties"]["action"]
    assert "evaluate" not in action_schema["enum"]
    assert "send_photo" not in schemas["browse_page"]["description"]
    assert "analyze_screenshot" in schemas["browse_page"]["description"]
    assert schemas["browse_page"]["parameters"]["properties"]["engine"]["enum"] == ["chromium", "webkit"]
    assert "device" in schemas["browse_page"]["parameters"]["properties"]
    assert list_non_core_tools(registry) == []


def test_workspace_parent_keeps_the_ordinary_top_level_control_surface(tmp_path, monkeypatch):
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.mcp_client as mcp_client

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    ))

    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda *args, **kwargs: None)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: type("_M", (), {"list_tools_for_registry": lambda self: []})())
    names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}

    assert "plan_task" in names
    assert "task_acceptance_review" in names
    assert "commit_reviewed" in names
    assert "request_restart" in names

    registry.override_handler("task_acceptance_review", lambda ctx=None, **_kwargs: "review-ok")
    registry.override_handler("commit_reviewed", lambda ctx=None, **_kwargs: "commit-ok")
    assert registry.execute("task_acceptance_review", {}) == "review-ok"
    assert registry.execute("commit_reviewed", {"commit_message": "system target"}) == "commit-ok"


def test_workspace_focus_does_not_turn_top_level_cancel_into_child_only(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools import join_ledger
    from ouroboros.tools.registry import ToolContext

    system, workspace, data = tmp_path / "system", tmp_path / "workspace", tmp_path / "data"
    for path in (system, workspace, data):
        path.mkdir()
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="parent",
    )
    monkeypatch.setattr(join_ledger, "_is_own_child", lambda *_a, **_k: False)
    # NB: no ``write_task_result`` patch — the cancel tool no longer writes a
    # status latch at all (phase A: it records a durable cancel INTENT), and the
    # symbol is not imported here any more, so patching it raised AttributeError.
    monkeypatch.setattr("ouroboros.tools.control._emit_control_event", lambda *_a, **_k: "live")

    assert join_ledger._cancel_task(ctx, "foreign-task").startswith("Cancel requested")

    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    assert "may only cancel its own children" in join_ledger._cancel_task(ctx, "foreign-task")


def test_schedule_subagent_inherits_workspace_executor_ref(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from tests._shared import configure_test_subagent

    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True)
    task_contract = build_task_contract({
        "resource_policy": {
            "protected_artifacts": [
                {
                    "id": "reference",
                    "role": "black_box_reference",
                    "paths": ["/workspace/executable"],
                    "allow": ["execute"],
                }
            ]
        }
    })
    executor_ref = {
        "type": "docker_exec",
        "id": "pb-container",
        "container_name": "pb-container",
        "network": "none",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="parent-task",
        task_contract=task_contract,
        task_metadata={"task_contract": task_contract},
        executor_ref=executor_ref,
    )
    registry.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    subagent_id = configure_test_subagent(monkeypatch)

    result = registry.execute(
        "schedule_subagent",
        {
            "subagent_id": subagent_id,
            "objective": "Inspect the workspace contract.",
            "expected_output": "A concise report.",
            "role": "auditor",
        },
    )

    assert "Subagent request queued" in result
    assert ctx.pending_events
    event = ctx.pending_events[0]
    assert event["executor_ref"] == executor_ref
    assert event["metadata"]["executor_ref"] == executor_ref
    child_id = event["task_id"]
    persisted = json.loads((data / "task_results" / f"{child_id}.json").read_text(encoding="utf-8"))
    assert persisted["executor_ref"] == executor_ref
    assert persisted["task_contract"]["resource_policy"]["protected_artifacts"][0]["paths"] == ["/workspace/executable"]


def test_capability_omission_manifest_surfaces_extension_discovery_failure(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.tools import tool_discovery
    from ouroboros.tools.registry import ToolRegistry

    class BoomLock:
        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    monkeypatch.setattr(extension_loader, "_lock", BoomLock())

    registry.schemas()
    tool_discovery.set_registry(registry)
    text = tool_discovery._list_available_tools(registry._ctx)

    assert "CAPABILITY_OMISSION_MANIFEST" in text
    assert "extensions" in text
    assert "boom" in text
