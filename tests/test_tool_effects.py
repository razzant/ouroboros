"""Effect authority is the published generation, never editable skill prose or MCP hints."""
from types import SimpleNamespace

import pytest

from ouroboros import extension_registry_state as registry, tool_effects as effects
from ouroboros.extension_surface_names import extension_surface_name


@pytest.fixture
def published_tool(monkeypatch):
    name = extension_surface_name("connector", "operate")
    descriptor = {"skill": "connector", "extension_generation": "generation-one"}
    manifest = SimpleNamespace(raw_extra={"tools": []})
    bundle = registry._ExtensionRegistrations(generation_digest="generation-one", loaded_skill=SimpleNamespace(manifest=manifest))
    monkeypatch.setitem(registry._tools, name, descriptor)
    monkeypatch.setitem(registry._extensions, "connector", bundle)

    def publish(effect="read", **kwargs):
        manifest.raw_extra["tools"] = [{"name": "operate", "effect": effect, **kwargs}]
        return name, descriptor

    publish.bundle = bundle
    return publish


def test_published_manifest_only_and_generation_match(published_tool, tmp_path, monkeypatch):
    name, descriptor = published_tool("message", target=["to", "cc", "bcc"], conversation="room", thread="thread")
    published_tool.bundle.skill_dir = str(tmp_path)
    (tmp_path / "SKILL.md").write_text("tools:\n- name: operate\n  effect: read\n")
    monkeypatch.setattr("ouroboros.skill_loader.load_skill", lambda *a, **k: pytest.fail("mutable manifest read"))
    fact = effects.effect_of(name, ext_tool=descriptor)
    assert (fact.effect, fact.declared, fact.targets, fact.conversation, fact.thread) == (
        "message", True, ("to", "cc", "bcc"), "room", "thread")
    assert effects.effect_of(name) == fact  # Safety has no descriptor argument.
    published_tool.bundle.generation_digest = "generation-two"
    published_tool.bundle.loaded_skill.manifest.raw_extra["tools"][0]["effect"] = "read"
    assert effects.effect_of(name, ext_tool=descriptor).effect == "write"
    assert not effects.effect_of(name, ext_tool=descriptor).declared
    descriptor["extension_generation"] = "generation-two"
    assert effects.effect_of(name, ext_tool=descriptor).effect == "read"


@pytest.mark.parametrize("value,expected,declared", [
    ("read", "read", True), ("message", "message", True), ("write", "write", True),
    (None, "write", False), ("mixed", "write", False), (["read", "write"], "write", False),
])
def test_effect_defaults_and_mixed_tools(published_tool, value, expected, declared):
    name, descriptor = published_tool(value)
    assert (effects.effect_of(name, ext_tool=descriptor).effect, effects.effect_of(name).declared) == (expected, declared)


def test_absent_duplicate_and_builtin(published_tool):
    name, descriptor = published_tool()
    assert effects.effect_of(name).declared
    rows = published_tool.bundle.loaded_skill.manifest.raw_extra["tools"]
    rows.append(dict(rows[0]))
    assert not effects.effect_of(name).declared
    rows.clear()
    assert effects.effect_of(name, ext_tool=descriptor).effect == "write"
    assert effects.effect_of("read_file") is effects.BUILTIN


def test_mcp_annotations_are_retained_without_authority(monkeypatch):
    monkeypatch.setattr(effects, "_MCP_ANNOTATIONS", {})
    effects.record_mcp_annotations("example", [{"name": "read", "annotations": {
        "readOnlyHint": True, "destructiveHint": False, "title": "not retained"}}])
    fact = effects.effect_of("mcp_example__read")
    assert fact.effect == "write" and not fact.declared
    assert fact.annotations == {"readOnlyHint": True, "destructiveHint": False}
    effects.record_mcp_annotations("example", [])
    assert effects.effect_of("mcp_example__read", is_mcp=True).annotations == {}
    assert effects.effect_of("mcp_example__read").effect == "write"


def test_target_values_are_flat_declared_arguments_only():
    fact = effects.ToolEffect("message", targets=("to", "cc", "bcc"), conversation="room", thread="thread")
    assert effects.target_values(fact, {"to": ["a", " b "], "cc": 123, "bcc": {"nested": "c"},
                                       "room": False, "thread": "T", "undeclared": "d"}) == {
        "to": ["a", "b"], "cc": ["123"], "thread": ["T"]}
