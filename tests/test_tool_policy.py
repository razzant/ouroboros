"""Tests for task-start tool visibility policy."""

import inspect
import pathlib
import tempfile

import ouroboros.loop as loop_mod
from ouroboros.tool_capabilities import CORE_TOOL_NAMES
from ouroboros.tool_policy import initial_tool_schemas, list_non_core_tools
from ouroboros.tools.registry import ToolRegistry


def _build_registry() -> ToolRegistry:
    tmp = pathlib.Path(tempfile.mkdtemp())
    return ToolRegistry(repo_dir=tmp, drive_root=tmp)


def test_core_surface_includes_user_message_and_media():
    assert "send_photo" in CORE_TOOL_NAMES
    assert "send_video" in CORE_TOOL_NAMES
    assert "send_user_message" in CORE_TOOL_NAMES
    assert "send_links" in CORE_TOOL_NAMES


def test_initial_tool_schemas_include_media_and_meta_tools():
    registry = _build_registry()
    names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}
    assert "send_photo" in names
    assert "send_links" in names
    assert "list_available_tools" in names
    assert "enable_tools" in names


def test_non_core_listing_excludes_core_media_tools():
    registry = _build_registry()
    names = {entry["name"] for entry in list_non_core_tools(registry)}
    assert "send_photo" not in names
    assert "send_links" not in names
    assert "plan_task" not in names


def test_loop_bootstraps_from_tool_policy():
    source = inspect.getsource(loop_mod)
    assert "initial_tool_schemas(tools, context_mode=active_context_mode)" in source
    assert "schemas(core_only=True)" not in source


def test_advisory_tools_in_core_tool_names():
    """preflight_review (and its callable advisory_review alias) and
    review_status must be core tools."""
    assert "preflight_review" in CORE_TOOL_NAMES
    assert "advisory_review" in CORE_TOOL_NAMES  # compat alias stays core-callable
    assert "review_status" in CORE_TOOL_NAMES


def test_advisory_tools_in_initial_schemas():
    """preflight_review and review_status must appear in initial tool schemas;
    the advisory_review compat alias is callable but never advertised."""
    registry = _build_registry()
    names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}
    assert "preflight_review" in names
    assert "advisory_review" not in names
    assert "review_status" in names


def test_heal_skill_tools_are_core_visible_in_initial_schemas():
    """Heal prompts must be able to call skill_review and
    skill_preflight without enable_tools (enable_tools is blocked in heal
    mode). Pin both the tool_policy SSOT and registry.core_only fallback."""
    assert "skill_review" in CORE_TOOL_NAMES
    assert "skill_preflight" in CORE_TOOL_NAMES
    registry = _build_registry()
    names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}
    assert "skill_review" in names
    assert "skill_preflight" in names
    core_only_names = {schema["function"]["name"] for schema in registry.schemas(core_only=True)}
    assert "skill_review" in core_only_names
    assert "skill_preflight" in core_only_names


def test_enable_tools_does_not_duplicate_active_tool_schemas():
    registry = _build_registry()
    tool_schemas = initial_tool_schemas(registry)
    registry._capability_omissions = [{"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"}]
    messages = []
    tool_schemas, _enabled_extra = loop_mod._setup_dynamic_tools(registry, tool_schemas, messages)
    assert any("[CAPABILITY_OMISSION_MANIFEST]" in str(message.get("content") or "") for message in messages)
    assert all(message.get("role") != "system" for message in messages)

    core_result = registry.execute("enable_tools", {"tools": "preflight_review"})
    names_after_core = [schema["function"]["name"] for schema in tool_schemas]
    assert names_after_core.count("preflight_review") == 1
    assert "already active" in core_result

    extra_result = registry.execute("enable_tools", {"tools": "plan_task"})
    names_after_extra = [schema["function"]["name"] for schema in tool_schemas]
    assert names_after_extra.count("plan_task") == 1
    assert "already active" in extra_result

    extra_again_result = registry.execute("enable_tools", {"tools": "plan_task"})
    names_after_extra_again = [schema["function"]["name"] for schema in tool_schemas]
    assert names_after_extra_again.count("plan_task") == 1
    assert "already active" in extra_again_result


def test_nano_initial_view_uses_compact_schema_selection():
    registry = _build_registry()
    max_names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}
    nano_names = {
        schema["function"]["name"]
        for schema in initial_tool_schemas(registry, context_mode="nano")
    }
    assert nano_names
    assert nano_names <= max_names
    assert len(nano_names) < len(max_names)


def test_list_available_tools_lists_names_when_everything_is_resident():
    """#1262: residency is an attribute of a row, never the answer. A Max task
    whose whole catalog is loaded still gets the callable names back."""
    registry = _build_registry()
    tool_schemas = initial_tool_schemas(registry)
    messages = []
    loop_mod._setup_dynamic_tools(registry, tool_schemas, messages)

    before = registry.execute("list_available_tools", {})
    assert "All tools are already" not in before
    assert "- builtin:" in before and "Built-in, loaded:" in before
    assert "plan_task" in before and "read_file" in before
    assert "not loaded" not in before.split("Namespaces")[1]

    # A schema dropped from the resident list is listed as not loaded (with its
    # purpose once the namespace is selected); enabling it makes it resident
    # again without duplicating it.
    tool_schemas[:] = [s for s in tool_schemas if s["function"]["name"] != "plan_task"]
    assert "Built-in, not loaded: plan_task" in registry.execute("list_available_tools", {})
    assert "- plan_task [not loaded]: " in registry.execute("list_available_tools", {"namespace": "builtin"})
    assert "registered late" in registry.execute("enable_tools", {"tools": "plan_task"})
    after = registry.execute("list_available_tools", {"namespace": "builtin"})
    assert "- plan_task [loaded]" in after
    assert [s["function"]["name"] for s in tool_schemas].count("plan_task") == 1


def test_live_extension_tools_are_initial_not_non_core(monkeypatch):
    from ouroboros import extension_loader

    registry = _build_registry()
    tool_name = extension_loader.extension_surface_name("weather", "forecast")
    with extension_loader._lock:
        extension_loader._tools[tool_name] = {
            "name": tool_name,
            "handler": lambda ctx: "ok",
            "description": "Forecast",
            "schema": {"type": "object", "properties": {}},
            "timeout_sec": 5,
            "skill": "weather",
        }
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    try:
        initial_names = {schema["function"]["name"] for schema in initial_tool_schemas(registry)}
        non_core_names = {entry["name"] for entry in list_non_core_tools(registry)}
        assert tool_name in initial_names
        assert tool_name not in non_core_names
    finally:
        with extension_loader._lock:
            extension_loader._tools.pop(tool_name, None)


def test_list_skills_is_core_visible_for_repair(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    names = {schema.get("name") or schema.get("function", {}).get("name") for schema in registry.schemas(core_only=True)}
    assert "list_skills" in names
