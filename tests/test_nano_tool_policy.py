"""Nano schema residency preserves capabilities, canonical bytes and task isolation."""

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.tool_policy import (
    NANO_SCHEMA_META_NAMES,
    compact_tool_catalog,
    format_capability_omissions,
    initial_tool_schemas,
    list_non_core_tools,
    request_tool_schema_selection,
    select_tool_schemas,
)


def _schema(name, description=None):
    return {"type": "function", "function": {
        "name": name, "description": description or f"Use {name} for its stated purpose.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }}


class Registry:
    def __init__(self):
        self.canonical = [_schema(name) for name in (
            "read_file", "list_available_tools", "write_file", "enable_tools", "compact_context",
        )]

    def schemas(self):
        return self.canonical

    def policy_hidden_reason(self, name):
        return {"blocked": "disabled by this task's contract"}.get(name)


def _names(schemas):
    return tuple(schema["function"]["name"] for schema in schemas)


def test_only_nano_selects_schemas_and_never_changes_shared_canonical_bytes():
    registry = Registry()
    before = json.dumps(registry.schemas())
    full = initial_tool_schemas(registry)
    nano = initial_tool_schemas(registry, "nano", ["write_file"])
    assert _names(nano) == ("list_available_tools", "write_file", "enable_tools", "compact_context")
    for mode in ("low", "max"):
        assert initial_tool_schemas(registry, mode, []) == full
        assert list_non_core_tools(registry, mode) == []
    nano[1]["function"]["parameters"]["properties"]["path"]["type"] = "integer"
    assert json.dumps(registry.schemas()) == before
    assert json.dumps(initial_tool_schemas(registry, "max")) == before


def test_parent_selection_is_not_an_allowlist_and_meta_transport_survives_replacement():
    registry, ctx = Registry(), SimpleNamespace()
    current = initial_tool_schemas(registry, "nano", ["read_file"])
    request = request_tool_schema_selection(ctx, registry, ["write_file"],
                                            context_mode="nano", current_schemas=current)
    assert request.status == "pending"
    assert "write_file" in request.selection.chosen and "read_file" not in request.selection.chosen
    assert NANO_SCHEMA_META_NAMES <= set(request.selection.chosen)
    assert _names(current) == ("read_file", "list_available_tools", "enable_tools", "compact_context")
    assert ctx._pending_tool_schema_names == request.selection.chosen
    assert "write_file" in _names(registry.schemas())


def test_two_enable_requests_compose_pending_selection_and_no_op_keeps_bytes():
    registry, ctx, sibling = Registry(), SimpleNamespace(), SimpleNamespace()
    current = initial_tool_schemas(registry, "nano")
    first = request_tool_schema_selection(ctx, registry, ["read_file"], context_mode="nano",
                                         current_schemas=current, extend=True)
    second = request_tool_schema_selection(ctx, registry, ["write_file"], context_mode="nano",
                                          current_schemas=current, extend=True)
    assert set(first.selection.chosen) < set(second.selection.chosen)
    assert not hasattr(sibling, "_pending_tool_schema_names")
    repeated = request_tool_schema_selection(ctx, registry, ["write_file"], context_mode="nano",
                                            current_schemas=current, extend=True)
    assert repeated.selection == second.selection
    # The loop owns the real completed-tool-boundary apply; these are its bytes.
    applied = list(second.selection.schemas)
    no_op = request_tool_schema_selection(ctx, registry, second.selection.chosen,
                                         context_mode="nano", current_schemas=applied)
    assert no_op.status == "no_op"
    assert json.dumps(list(no_op.selection.schemas)) == json.dumps(applied)
    assert ctx._pending_tool_schema_names is None


def test_unavailable_unknown_and_missing_meta_are_facts_without_a_grant():
    registry, ctx = Registry(), SimpleNamespace()
    registry.canonical = [s for s in registry.canonical if s["function"]["name"] != "enable_tools"]
    request = request_tool_schema_selection(ctx, registry, ["blocked", "not_registered"],
                                            context_mode="nano", current_schemas=[])
    assert request.selection.unavailable == {"blocked": "disabled by this task's contract"}
    assert request.selection.unknown == ("not_registered",)
    assert request.selection.missing_meta == ("enable_tools",)
    assert set(request.selection.chosen) == {"list_available_tools", "compact_context"}
    assert "enable_tools" not in _names(registry.schemas())


def test_catalog_has_every_permitted_name_and_only_purpose_is_bounded():
    canonical = [_schema("tool_with_a_long_name_" + "x" * 150, "purpose " * 100), _schema("other")]
    before = copy.deepcopy(canonical)
    rows = compact_tool_catalog(canonical, schema_names=["other"])
    assert [row["name"] for row in rows] == list(_names(canonical))
    assert rows[0]["description_truncated"] is True
    assert rows[0]["residency"] == "not_loaded"
    assert rows[1]["residency"] == "loaded"
    assert canonical == before
    omissions = [{"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"}]
    assert format_capability_omissions(omissions)[1] == "- mcp: resource_blocked (network=false)"


@pytest.mark.parametrize("mode", ["low", "max"])
def test_full_modes_do_not_queue_a_selection_or_retain_old_nano_pending(mode):
    registry, ctx = Registry(), SimpleNamespace(_pending_tool_schema_names=("read_file",))
    request = request_tool_schema_selection(ctx, registry, [], context_mode=mode,
                                            current_schemas=initial_tool_schemas(registry))
    assert request.status == "full_envelope_unchanged"
    assert _names(request.selection.schemas) == _names(registry.schemas())
    assert ctx._pending_tool_schema_names is None


def test_actual_registry_dynamic_tools_keep_canonical_order_and_permissions(tmp_path, monkeypatch):
    from ouroboros import extension_loader, mcp_client
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    name = extension_loader.extension_surface_name("weather", "forecast")
    with extension_loader._lock:
        monkeypatch.setitem(extension_loader._tools, name, {
            "name": name, "handler": lambda ctx: "ok", "description": "Forecast weather.",
            "schema": {"type": "object", "properties": {}}, "timeout_sec": 5, "skill": "weather",
        })
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **kwargs: None)
    mcp_rows = [{"name": "mcp_example__lookup", "description": "Lookup records.", "schema": {"type": "object", "properties": {}}}]
    monkeypatch.setattr(mcp_client, "get_manager", lambda: SimpleNamespace(
        list_tools_for_registry=lambda: mcp_rows, enabled_servers_without_tools=lambda: [],
        catalog_notices=lambda: [],
    ))
    full = registry.schemas()
    selected = initial_tool_schemas(registry, "nano", ["mcp_example__lookup", name])
    assert _names(selected) == tuple(n for n in _names(full) if n in NANO_SCHEMA_META_NAMES or n in (name, "mcp_example__lookup"))
    # An unpinned MCP server starts on demand in every mode; a pinned one keeps its schemas (#2332).
    assert initial_tool_schemas(registry, "max") == [s for s in full if s["function"]["name"] != "mcp_example__lookup"]
    mcp_rows[0]["schemas_at_start"] = True
    assert initial_tool_schemas(registry, "max") == full
    mcp_rows[0]["schemas_at_start"] = False
    assert [r["name"] for r in compact_tool_catalog(full, schema_names=_names(selected))] == list(_names(full))
    mcp_rows.append({"name": "mcp_example__later", "description": "Added after startup.", "schema": {"type": "object", "properties": {}}})
    request = request_tool_schema_selection(registry._ctx, registry, ["mcp_example__later"],
                                            context_mode="nano", current_schemas=selected, extend=True)
    assert request.selection.chosen[-1] == "mcp_example__later"
    registry._ctx.task_contract = {"disabled_tools": [name, "mcp_example__later"]}
    refused = request_tool_schema_selection(registry._ctx, registry, [name, "mcp_example__later"],
                                            context_mode="nano", current_schemas=selected)
    assert set(refused.selection.unavailable) == {name, "mcp_example__later"}
    assert name not in refused.selection.chosen and "mcp_example__later" not in refused.selection.chosen


def test_pure_selection_rejects_a_string_instead_of_interpreting_its_characters():
    with pytest.raises(ValueError, match="collection"):
        select_tool_schemas(Registry().schemas(), context_mode="nano", schema_names="read_file")
