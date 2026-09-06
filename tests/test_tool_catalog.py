"""Contracts for the immutable built-in catalog and dynamic name precedence."""

from __future__ import annotations

import dataclasses
import importlib
import typing
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader, mcp_client
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.tool_policy import format_capability_omissions
from ouroboros.tools import registry_core
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.tool_catalog import (
    DuplicateToolNameError,
    ToolCatalog,
    ToolEntry,
)
from ouroboros.tools.tool_result import ToolResult
from tests.test_extension_loader import _prepare_extension


def _default_handler(_ctx, **_kwargs):
    return "ok"


def _entry(name: str, handler=None) -> ToolEntry:
    return ToolEntry(
        name=name,
        schema={
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler or _default_handler,
    )


def _schema_names(registry: ToolRegistry, *, core_only: bool = False) -> list[str]:
    return [
        str(item.get("function", {}).get("name") or "")
        for item in registry.schemas(core_only=core_only)
    ]


def test_tool_entry_is_shallow_frozen():
    schema = {"name": "demo"}
    entry = ToolEntry("demo", schema, lambda _ctx: "ok")

    assert typing.get_type_hints(ToolEntry) == {
        "name": str,
        "schema": typing.Dict[str, typing.Any],
        "handler": typing.Callable,
        "is_code_tool": bool,
        "timeout_sec": int,
        "mutates_worktree": bool,
        # tip drift: upstream added the compat-alias field after the reference
        # cutoff (pinned in tests/test_tool_owner_facades.py as well).
        "alias_for": str,
    }
    fields = {item.name: item for item in dataclasses.fields(ToolEntry)}
    assert fields["is_code_tool"].default is False
    assert type(fields["is_code_tool"].default) is bool
    assert fields["timeout_sec"].default == 360
    assert type(fields["timeout_sec"].default) is int
    assert fields["mutates_worktree"].default is False
    assert type(fields["mutates_worktree"].default) is bool

    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.name = "changed"
    schema["description"] = "still ABI-compatible"
    assert entry.schema["description"] == "still ABI-compatible"


def test_tool_catalog_is_immutable_and_reports_both_duplicate_origins():
    first = _entry("same")
    catalog = ToolCatalog([("alpha.get_tools[0]", first)])

    assert catalog.entries["same"] is first
    assert catalog.origin_for("same") == "alpha.get_tools[0]"
    with pytest.raises(TypeError):
        catalog.entries["new"] = _entry("new")
    with pytest.raises(DuplicateToolNameError) as caught:
        ToolCatalog([
            ("alpha.get_tools[0]", first),
            ("alpha.get_tools[1]", _entry("same")),
        ])
    assert caught.value.first_origin == "alpha.get_tools[0]"
    assert caught.value.duplicate_origin == "alpha.get_tools[1]"


def test_registry_loader_does_not_degrade_a_first_party_duplicate(
    tmp_path, monkeypatch,
):
    module = SimpleNamespace(get_tools=lambda: [_entry("same"), _entry("same")])
    real_import = importlib.import_module

    # v7next adaptation: this tree's loader discovers modules via the frozen
    # list / pkgutil (no tool_modules_for_runtime inventory leaf here); pin the
    # discovery to the one duplicate module through the frozen branch.
    import sys as _sys

    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        registry_core.ToolRegistry, "_FROZEN_TOOL_MODULES", ["duplicate"],
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module if name == "ouroboros.tools.duplicate" else real_import(name),
    )

    with pytest.raises(DuplicateToolNameError) as caught:
        ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    assert caught.value.first_origin.endswith("duplicate.get_tools[0]")
    assert caught.value.duplicate_origin.endswith("duplicate.get_tools[1]")


def test_scoped_registration_isolated_from_base_and_sibling_registry(tmp_path):
    first = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    second = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)

    def scoped_handler(_ctx):
        return "scoped"

    scoped = _entry("scoped_demo", scoped_handler)
    first.register(scoped)
    assert first._entries["scoped_demo"] is scoped
    assert "scoped_demo" not in first._base_catalog.entries
    assert "scoped_demo" not in second._entries
    assert first._entry_origins["scoped_demo"].endswith("scoped_handler")

    with pytest.raises(DuplicateToolNameError) as base_collision:
        first.register(_entry("read_file"), origin="task.fixture.read_file")
    assert base_collision.value.first_origin.endswith("core.get_tools[0]")
    assert base_collision.value.duplicate_origin == "task.fixture.read_file"

    def duplicate_handler(_ctx):
        return "duplicate"

    with pytest.raises(DuplicateToolNameError) as caught:
        first.register(_entry("scoped_demo", duplicate_handler))
    assert caught.value.first_origin.endswith("scoped_handler")
    assert caught.value.duplicate_origin.endswith("duplicate_handler")


def test_handler_override_is_an_overlay_not_a_base_catalog_mutation(
    tmp_path, monkeypatch,
):
    first = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    second = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    base_entry = first._base_catalog.entries["read_file"]

    def replacement(_ctx, **_kwargs):
        return "replacement"

    first.override_handler("read_file", replacement)

    assert first._entries["read_file"].handler is replacement
    assert first._handler_overrides["read_file"] is replacement
    assert first._base_catalog.entries["read_file"] is base_entry
    assert base_entry.handler is not replacement
    assert second._entries["read_file"].handler is not replacement

    first.register(_entry("scoped_override"))
    first.override_handler("scoped_override", replacement)
    import ouroboros.safety as safety_mod

    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **kw: (True, ""))
    assert first.execute("scoped_override", {}) == "replacement"
    entries_before = dict(first._entries)
    first.override_handler("unknown_override", replacement)
    assert first._entries == entries_before


def test_extension_collision_keeps_catalog_entry_and_is_visible(
    tmp_path, monkeypatch, caplog,
):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    loaded, _, drive_root = _prepare_extension(
        tmp_path,
        "catalog_collision",
        (
            "def _dynamic(ctx): return 'dynamic'\n"
            "def register(api):\n"
            "    api.register_tool('echo', _dynamic, description='dynamic', "
            "schema={'type': 'object', 'properties': {}}, timeout_sec=1)\n"
        ),
        permissions=["tool"],
    )
    error = extension_loader.load_extension(
        loaded, lambda: {}, drive_root=drive_root, _force_in_process=True,
    )
    tool_name = extension_loader.extension_surface_name(
        "catalog_collision", "echo",
    )
    try:
        assert error is None
        dynamic = extension_loader.get_tool(tool_name)
        assert dynamic is not None
        registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
        registry.register(_entry(tool_name), origin="task.extension_collision")
        with caplog.at_level("ERROR"):
            names = _schema_names(registry)
        assert names.count(tool_name) == 1
        import ouroboros.safety as safety_mod

        monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **kw: (True, ""))
        assert registry.execute(tool_name, {}) == "ok"
        assert extension_loader.get_tool(tool_name) == dynamic
        assert tool_name in extension_loader.snapshot()["tools"]
        assert "extensions tool name collision omitted" in caplog.text
        assert registry.get_timeout(tool_name) == registry._entries[tool_name].timeout_sec
        omissions = registry.capability_omissions()
        assert any(
            item.get("surface") == "extensions"
            and item.get("reason") == "name_collision"
            and item.get("tools") == [tool_name]
            for item in omissions
        )
        assert any(
            tool_name in line
            for line in format_capability_omissions(omissions)
        )

        contract = build_task_contract({
            "disabled_tools": [tool_name],
            "allowed_resources": {"network": True},
        })
        registry.set_context(
            ToolContext(
                repo_dir=tmp_path,
                drive_root=drive_root,
                task_contract=contract,
                task_metadata={"task_contract": contract},
            )
        )
        assert tool_name not in _schema_names(registry)
        assert any(
            item.get("surface") == "extensions"
            and item.get("reason") == "name_collision"
            for item in registry.capability_omissions()
        )
    finally:
        extension_loader.unload_extension("catalog_collision")
    assert tool_name not in extension_loader.snapshot()["tools"]


def test_extension_collision_is_not_disclosed_without_an_acting_grant(
    tmp_path, monkeypatch,
):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    with extension_loader._lock:
        previous = extension_loader._tools.get("read_file")
        extension_loader._tools["read_file"] = {
            "name": "read_file",
            "skill": "private-collision",
            "schema": {"type": "object", "properties": {}},
    }
    try:
        for grants, visible in (((), False), (("read_file",), True)):
            registry.set_context(
                ToolContext(
                    repo_dir=tmp_path,
                    drive_root=tmp_path,
                    task_constraint=TaskConstraint(
                        mode="acting_subagent",
                        surface="self_worktree",
                        write_root=str(tmp_path),
                        external_tool_grants=grants,
                    ),
                )
            )
            registry.schemas(core_only=True)
            collisions = [
                item for item in registry.capability_omissions()
                if item.get("reason") == "name_collision"
            ]
            assert bool(collisions) is visible
    finally:
        with extension_loader._lock:
            if previous is None:
                extension_loader._tools.pop("read_file", None)
            else:
                extension_loader._tools["read_file"] = previous


def test_mcp_collision_keeps_catalog_schema_and_timeout(
    tmp_path, monkeypatch, caplog,
):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.register(_entry("mcp_demo__foo"), origin="task.mcp_demo__foo")
    manager = mcp_client.MCPManager()
    calls = []

    async def list_tools(_cfg, _timeout):
        return [{
            "name": "foo",
            "description": "colliding MCP tool",
            "input_schema": {"type": "object", "properties": {}},
        }]

    async def call_tool(cfg, name, arguments, timeout):
        calls.append((cfg.id, name, arguments, timeout))
        return ToolResult(
            status="ok",
            code="OK",
            text="dynamic",
            meta={"mcp_is_error": False},
        )

    manager._async_list_tools = list_tools
    manager._async_call_tool = call_tool
    manager.reconfigure({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 60,
        "MCP_SERVERS": [{
            "id": "demo",
            "enabled": True,
            "transport": "streamable_http",
            "url": "https://example.invalid/mcp",
            "allowed_tools": [],
        }],
    })
    assert manager.refresh_server("demo")["ok"] is True
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kwargs: None)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: manager)

    with caplog.at_level("ERROR"):
        names = _schema_names(registry)
    assert names.count("mcp_demo__foo") == 1
    import ouroboros.safety as safety_mod

    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **kw: (True, ""))
    assert registry.execute("mcp_demo__foo", {}) == "ok"
    assert calls == []
    assert manager.get_tool("mcp_demo__foo")["raw_name"] == "foo"
    assert manager.refresh_server("demo")["ok"] is True
    assert "mcp tool name collision omitted" in caplog.text
    assert registry.get_schema_by_name("mcp_demo__foo") == registry._schema_for_entry(
        registry._entries["mcp_demo__foo"]
    )
    assert registry.get_timeout("mcp_demo__foo") == registry._entries["mcp_demo__foo"].timeout_sec
    assert any(
        item.get("surface") == "mcp"
        and item.get("kind") == "registry_shadow"
        and item.get("tools") == ["mcp_demo__foo"]
        for item in registry.capability_omissions()
    )


def test_acting_mcp_collisions_are_visible_only_after_exact_grant(
    tmp_path, monkeypatch,
):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.register(_entry("mcp_demo__foo_bar"), origin="task.mcp_demo__foo_bar")

    class FakeManager:
        def list_tools_for_registry(self):
            return [{
                "name": "mcp_demo__foo_bar",
                "description": "colliding MCP tool",
                "schema": {"type": "object", "properties": {}},
                "server_id": "demo",
                "raw_name": "foo-bar",
            }]

        def tool_name_collisions(self):
            return [{
                "prefixed_name": "mcp_demo__foo_bar",
                "kept_raw_name": "foo-bar",
                "dropped_raw_name": "foo_bar",
                "server_id": "demo",
            }]

        def enabled_servers_without_tools(self):
            return []

    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kwargs: None)
    monkeypatch.setattr(mcp_client, "get_manager", lambda: FakeManager())

    for grants, expected_kinds in (
        ((), set()),
        (("mcp_demo__foo_bar",), {"registry_shadow", "provider_slug"}),
    ):
        registry.set_context(
            ToolContext(
                repo_dir=tmp_path,
                drive_root=tmp_path,
                task_constraint=TaskConstraint(
                    mode="acting_subagent",
                    surface="self_worktree",
                    write_root=str(tmp_path),
                    external_tool_grants=grants,
                ),
            )
        )
        registry.schemas()
        collisions = [
            item for item in registry.capability_omissions()
            if item.get("reason") == "name_collision"
        ]
        assert {item.get("kind") for item in collisions} == expected_kinds
