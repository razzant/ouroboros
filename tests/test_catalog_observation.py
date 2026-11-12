"""One look at the tool catalog reads the installed skills once.

Listing the catalog used to answer an extension's liveness once per extension TOOL, each answer
rediscovering every installed skill (a desktop skill's 24 tools meant 24 full manifest walks, twice per
`list_available_tools`): seconds per listing on an install with many skills. A skill's tools share one
verdict, one listing shares one skill inventory, and the next listing looks again.
"""
from __future__ import annotations


import pytest

import ouroboros.loop as loop_mod
from ouroboros import extension_loader
from ouroboros.tool_policy import initial_tool_schemas
from ouroboros.tools.registry import ToolRegistry


def _tools(skill: str, count: int) -> dict:
    rows = {}
    for index in range(count):
        name = extension_loader.extension_surface_name(skill, f"act{index}")
        rows[name] = {"name": name, "handler": lambda ctx: "ok", "description": f"{skill} action {index}",
                      "schema": {"type": "object", "properties": {}}, "timeout_sec": 5, "skill": skill}
    return rows


@pytest.fixture
def catalog(tmp_path):
    rows = {**_tools("deskx", 24), **_tools("kee", 3)}
    with extension_loader._lock:
        saved = dict(extension_loader._tools)
        extension_loader._tools.update(rows)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    try:
        yield registry, rows
    finally:
        with extension_loader._lock:
            extension_loader._tools.clear()
            extension_loader._tools.update(saved)


def _names(registry) -> set:
    return {schema["function"]["name"] for schema in registry.schemas()}


def test_a_skills_tools_share_one_liveness_verdict(catalog, monkeypatch):
    registry, rows = catalog
    asked = []
    monkeypatch.setattr(extension_loader, "is_extension_live",
                        lambda skill, root, repo_path=None: asked.append(skill) or skill != "kee")
    names = _names(registry)
    assert sorted(asked) == ["deskx", "kee"]  # once per skill, not 27 times
    assert {name for name in rows if rows[name]["skill"] == "deskx"} <= names
    assert not {name for name in rows if rows[name]["skill"] == "kee"} & names  # a dead skill's tools are absent


def test_a_skills_name_lists_its_namespace_and_the_overview_names_the_skill(catalog, monkeypatch):
    registry, rows = catalog
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    loop_mod._setup_dynamic_tools(registry, initial_tool_schemas(registry, context_mode="nano"), [], context_mode="nano")
    namespace = extension_loader.extension_name_prefix("deskx")[:-1]
    by_skill = registry.execute("list_available_tools", {"namespace": "deskx"})
    by_namespace = registry.execute("list_available_tools", {"namespace": namespace})
    assert by_skill == by_namespace and f"{namespace}: 24 callable tools" in by_skill
    assert "No tool in nosuch is currently callable." in registry.execute("list_available_tools", {"namespace": "nosuch"})
    overview = registry.execute("list_available_tools", {})
    assert f"- {namespace} (skill deskx): 24 tools" in overview
    assert "(skill" not in overview.split("- builtin:")[1].splitlines()[0]  # built-in rows name no skill
