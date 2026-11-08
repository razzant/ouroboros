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
    assert effects.effect_of(name) == fact  # Unbound lookup agrees while the generation is current.
    published_tool.bundle.generation_digest = "generation-two"
    published_tool.bundle.loaded_skill.manifest.raw_extra["tools"][0]["effect"] = "read"
    assert effects.effect_of(name, ext_tool=descriptor).effect == "write"
    assert not effects.effect_of(name, ext_tool=descriptor).declared
    descriptor["extension_generation"] = "generation-two"
    assert effects.effect_of(name, ext_tool=descriptor).effect == "read"


def test_published_gesture_declares_message_argument(published_tool):
    name, descriptor = published_tool("gesture", target="channel", conversation="room", message="ts")
    fact = effects.effect_of(name, ext_tool=descriptor)
    assert (fact.effect, fact.declared, fact.targets, fact.conversation, fact.message) == (
        "gesture", True, ("channel",), "room", "ts")



def test_a_loader_descriptor_keeps_its_own_publications_declaration(published_tool):
    """``get_tool`` captures the declaration with the handler: a replacement changes only later reads."""
    name, raw = published_tool("message", target="channel", conversation="channel", message="ts")
    published_tool.bundle.contributions = [{"provides": "place", "id": "place"}]
    bound = registry.get_tool_stamped(name)
    published_tool.bundle.generation_digest = raw["extension_generation"] = "generation-two"
    published_tool.bundle.loaded_skill.manifest.raw_extra["tools"] = [{"name": "operate", "effect": "read"}]
    published_tool.bundle.contributions = []
    fact = effects.effect_of(name, ext_tool=bound)
    assert (fact.effect, fact.declared, fact.targets, fact.conversation, fact.message) == (
        "message", True, ("channel",), "channel", "ts")
    assert effects.bound_declaration(bound)["provides_place"] is True
    later = registry.get_tool_stamped(name)
    assert effects.effect_of(name, ext_tool=later).effect == "read"
    assert effects.bound_declaration(later)["provides_place"] is False


def test_a_declaration_naming_another_generation_is_never_trusted(published_tool):
    """A raw or child-catalog descriptor can carry another process's capture; only the host's own read counts."""
    name, raw = published_tool("write")
    raw[registry.REVIEWED_DECLARATION] = {"extension_generation": "child-generation", "provides_place": False,
                                          "tool": {"name": "operate", "effect": "read"}}
    assert effects.bound_declaration(dict(raw)) is None
    assert effects.effect_of(name, ext_tool=dict(raw)).effect == "write"
    assert effects.effect_of(name, ext_tool=registry.get_tool_stamped(name)).effect == "write"

@pytest.mark.parametrize("value,expected,declared", [
    ("read", "read", True), ("message", "message", True), ("gesture", "gesture", True), ("write", "write", True),
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


def test_only_an_mcp_read_only_hint_declares_a_read(monkeypatch):
    """The configured server's ``readOnlyHint: true`` is a declared read; false, absent or non-boolean stays write."""
    monkeypatch.setattr(effects, "_MCP_ANNOTATIONS", {})
    effects.record_mcp_annotations("example", [
        {"name": "read", "annotations": {"readOnlyHint": True, "destructiveHint": False, "title": "not retained"}},
        {"name": "send", "annotations": {"readOnlyHint": False}},
        {"name": "vague", "annotations": {"readOnlyHint": "true"}}, {"name": "bare"}])
    fact = effects.effect_of("mcp_example__read")
    assert (fact.effect, fact.declared, fact.source) == ("read", True, "mcp")
    assert fact.annotations == {"readOnlyHint": True, "destructiveHint": False}
    for name in ("mcp_example__send", "mcp_example__vague", "mcp_example__bare", "mcp_example__unlisted"):
        assert (effects.effect_of(name).effect, effects.effect_of(name).declared) == ("write", False), name
    effects.record_mcp_annotations("example", [])
    assert effects.effect_of("mcp_example__read", is_mcp=True).annotations == {}
    assert effects.effect_of("mcp_example__read").effect == "write"


def test_target_values_are_flat_declared_arguments_only():
    fact = effects.ToolEffect("message", targets=("to", "cc", "bcc"), conversation="room", thread="thread")
    assert effects.target_values(fact, {"to": ["a", " b "], "cc": 123, "bcc": {"nested": "c"},
                                       "room": False, "thread": "T", "undeclared": "d"}) == {
        "to": ["a", "b"], "cc": ["123"], "thread": ["T"]}


def test_a_generic_tool_declares_which_calls_deliver_words(published_tool):
    """A connector names the calls of its generic tool that deliver words (``sends.when``) and where they land
    (dotted selectors); every other call keeps the tool's own declaration, and a malformed one selects nothing."""
    name, descriptor = published_tool("write", target=[], sends={
        "when": {"path": ["chat.postMessage", "chat.update"]},
        "target": "body.channel", "conversation": "body.channel", "thread": "body.thread_ts"})
    send = {"path": "chat.postMessage", "body": {"channel": "C1", "thread_ts": "1.2", "text": "hi"}}
    fact = effects.effect_of(name, ext_tool=descriptor, args=send)
    assert (fact.effect, fact.declared, fact.targets, fact.conversation, fact.thread) == (
        "message", True, ("body.channel",), "body.channel", "body.thread_ts")
    assert effects.target_values(fact, send) == {"body.channel": ["C1"], "body.thread_ts": ["1.2"]}
    for args in ({"path": "reactions.add", "body": {"channel": "C1"}}, {"body": {"channel": "C1"}}, None):
        assert effects.effect_of(name, ext_tool=descriptor, args=args).effect == "write"
    name, descriptor = published_tool("write", sends={"when": {"path": 5}, "target": "channel"})
    assert effects.effect_of(name, ext_tool=descriptor, args={"path": 5, "channel": "C1"}).effect == "write"
