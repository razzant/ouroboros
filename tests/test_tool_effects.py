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


def _declares(published_tool, version="2.5"):
    """The publication's manifest declares this PluginAPI minimum (``None``: no field, the legacy 1.3)."""
    published_tool.bundle.loaded_skill.manifest.plugin_api = {"version": version, "capabilities": []}


def test_groups_name_exact_addresses_each_with_its_own_account_and_thread(published_tool):
    """A list pairs each id with the group's one thread: its own id when the thread argument is the conversation
    argument, never another id's; blanks and repeats name nothing new."""
    _declares(published_tool)
    name, descriptor = published_tool("message", target="room", conversation="room", thread="topic", touches=[
        {"conversation": "to", "effect": "write"},
        {"conversation": "emitted", "thread": "emitted", "effect": "message"},
        {"conversation": "folders", "effect": "write", "account": "mailbox"},
        {"conversation": "source", "thread": "source_topic", "effect": "read"}])
    fact = effects.effect_of(name, ext_tool=descriptor)
    assert fact.declared and fact.problem == "" and [g.conversation for g in fact.touches] == ["to", "emitted", "folders", "source"]
    addresses, problem = effects.touched_addresses(fact, {
        "account_id": "A", "to": ["p@x", " ", "p@x"], "emitted": ["E1", "E2"], "folders": "Drafts", "mailbox": "M",
        "source": "S", "source_topic": "T"})
    assert problem == ""
    assert [(a.effect, a.conversation, a.account, a.thread, a.argument) for a in addresses] == [
        ("write", "p@x", "A", "", "to"), ("message", "E1", "A", "E1", "emitted"), ("message", "E2", "A", "E2", "emitted"),
        ("write", "Drafts", "M", "", "folders"), ("read", "S", "A", "T", "source")]


@pytest.mark.parametrize("args,problem", [
    ({"account_id": ["A"], "to": "p@x"}, "'account_id' holds a list"),  # one account, never inferred from a list
    ({"to": "p@x", "topic": ["t1"]}, "'topic' holds a list"),  # a second argument names one thread, never a list
    ({"to": {"address": "p@x"}}, "'to' holds a dict"),
    ({"to": ["p@x", True]}, "'to' holds a bool"),
    ({"account_id": ["A"]}, ""), ({"to": None}, ""), ({"to": []}, ""), ({"to": ""}, ""), ({}, ""),
])
def test_another_shape_is_a_problem_and_absence_reaches_nothing(published_tool, args, problem):
    _declares(published_tool)
    name, descriptor = published_tool("write", touches=[{"conversation": "to", "thread": "topic", "effect": "write"}])
    addresses, found = effects.touched_addresses(effects.effect_of(name, ext_tool=descriptor), args)
    assert addresses == () and (found.startswith(problem) if problem else found == ""), found


@pytest.mark.parametrize("effect,row,version,problem", [
    ("message", {"touches": [{"conversation": "to", "effect": "write"}]}, "2.4", 'plugin_api at "2.5"'),
    ("message", {"touches": [{"conversation": "to", "effect": "write"}]}, None, "binds 1.3"),
    ("read", {"touches": [{"conversation": "to", "effect": "write"}]}, "2.5", "touches[0]: a write group cannot ride a read call"),
    ("write", {"touches": [{"conversation": "to", "effect": "message"}]}, "2.5", "a message group cannot ride a write call"),
    ("message", {"touches": [{"conversation": "to", "effect": "gesture"}]}, "2.5", "a gesture group cannot ride a message call"),
    ("write", {"touches": [{"conversation": "to", "effect": "write", "each": "x"}]}, "2.5", "may hold only"),
    ("write", {"touches": [{"effect": "write"}]}, "2.5", "needs conversation"),
    ("write", {"touches": [{"conversation": "to", "effect": "speak"}]}, "2.5", "needs effect"),
    ("write", {"touches": [{"conversation": ["to"], "effect": "write"}]}, "2.5", "each name one argument"),
    ("write", {"touches": {"conversation": "to", "effect": "write"}}, "2.5", "touches must be a list"),
    ("write", {"touches": [{"conversation": "to", "effect": "write", "match": "fuzzy"}]}, "2.5",
     "touches[0]: match is scope or exact or exact_casefold"),
    ("write", {"touches": [{"conversation": "to", "effect": "write", "match": ["exact"]}]}, "2.5", "match is scope"),
    # An extra constraint is never where words or a gesture land: they keep their conversation's attribution and lease.
    ("message", {"touches": [{"conversation": "to", "effect": "message", "match": "exact"}]}, "2.5",
     "touches[0]: match exact is for a write or read group"),
    ("gesture", {"touches": [{"conversation": "to", "effect": "gesture", "match": "exact_casefold"}]}, "2.5",
     "touches[0]: match exact_casefold is for a write or read group"),
    ("write", {"sends": {"when": {"path": "send"}, "target": "room",
                         "touches": [{"conversation": "to", "effect": "gesture"}]}}, "2.5",
     "sends.touches[0]: a gesture group cannot ride a message call"),
])
def test_an_invalid_declaration_is_never_trusted_in_part(published_tool, effect, row, version, problem):
    """Whichever call it makes, a tool whose groups cannot be trusted is an undeclared write naming the problem."""
    if version:
        _declares(published_tool, version)
    name, descriptor = published_tool(effect, target="room", conversation="room", **row)
    for args in (None, {"room": "C1", "to": "p@x"}, {"path": "send", "room": "C1"}):
        fact = effects.effect_of(name, ext_tool=descriptor, args=args)
        assert (fact.effect, fact.declared, fact.targets, fact.touches) == ("write", False, (), ()), fact
        assert problem in fact.problem, fact.problem


def test_a_groups_match_rides_into_each_of_its_addresses_unfolded(published_tool):
    """``match`` defaults to scope and an explicit exact mode on a write or read group reaches each address as written:
    the comparator folds nothing here, only the place matcher does."""
    _declares(published_tool)
    name, descriptor = published_tool("message", target=["to"], conversation="room", touches=[
        {"conversation": "to", "effect": "write", "match": "exact_casefold"}, {"conversation": "folder", "effect": "write"},
        {"conversation": "source", "effect": "read", "match": "exact"}])
    fact = effects.effect_of(name, ext_tool=descriptor)
    assert fact.problem == "" and [(group.conversation, group.match) for group in fact.touches] == [
        ("to", "exact_casefold"), ("folder", "scope"), ("source", "exact")]
    addresses, problem = effects.touched_addresses(fact, {"account_id": "A", "to": ["Bob@X"], "folder": "Drafts",
                                                          "source": "S1"})
    assert problem == "" and [(a.conversation, a.match) for a in addresses] == [
        ("Bob@X", "exact_casefold"), ("Drafts", "scope"), ("S1", "exact")]


def test_a_rows_groups_ride_its_own_effect_and_a_sends_blocks_the_calls_it_selects(published_tool):
    _declares(published_tool)
    name, descriptor = published_tool("write", target="label", touches=[{"conversation": "label", "effect": "write"}],
                                      sends={"when": {"path": "send"}, "target": "room", "conversation": "room",
                                             "touches": [{"conversation": "to", "effect": "write"}]})
    send = effects.effect_of(name, ext_tool=descriptor, args={"path": "send", "room": "C1"})
    other = effects.effect_of(name, ext_tool=descriptor, args={"path": "label", "label": "L"})
    assert (send.effect, [g.conversation for g in send.touches]) == ("message", ["to"])
    assert (other.effect, [g.conversation for g in other.touches]) == ("write", ["label"])


def test_a_bound_call_keeps_its_publications_minimum(published_tool):
    """The minimum is part of what a bound call was read with: a replacement declaring an older one changes later
    reads only."""
    _declares(published_tool)
    name, _raw = published_tool("write", touches=[{"conversation": "to", "effect": "write"}])
    bound = registry.get_tool_stamped(name)
    assert effects.bound_declaration(bound)["plugin_api"] == "2.5"
    published_tool.bundle.generation_digest = _raw["extension_generation"] = "generation-two"
    _declares(published_tool, "2.4")
    assert effects.effect_of(name, ext_tool=bound).problem == ""
    assert "binds 2.4" in effects.effect_of(name, ext_tool=registry.get_tool_stamped(name)).problem


def test_thread_pairs_never_multiply_two_lists():
    assert effects.thread_pairs(["C"], ["t1", "t2"], same_argument=False) == ([("C", "t1"), ("C", "t2")], "")
    assert effects.thread_pairs(["C1", "C2"], ["t"], same_argument=False) == ([("C1", "t"), ("C2", "t")], "")
    assert effects.thread_pairs(["C1", "C2"], [], same_argument=True) == ([("C1", "C1"), ("C2", "C2")], "")
    pairs, problem = effects.thread_pairs(["C1", "C2"], ["t1", "t2"], same_argument=False)
    assert pairs == [] and "cannot be paired" in problem


@pytest.mark.parametrize("args,problem", [
    ({}, ""), ({"envelope": None}, ""), ({"envelope": {}}, ""), ({"envelope": {"to": None}, "box": 7}, ""),
    ({"envelope": "p@x"}, "'envelope' holds a str, so 'envelope.to' cannot be read"),
    ({"envelope": ["p@x"]}, "'envelope' holds a list, so 'envelope.to' cannot be read"),
    ({"envelope": {"to": "p@x"}, "box": 7}, "'box' holds a int, so 'box.account' cannot be read"),
    ({"envelope": {"to": "p@x"}, "box": {"account": ["M"]}}, "'box.account' holds a list"),
])
def test_a_supplied_path_that_cannot_be_read_is_never_absent(published_tool, args, problem):
    """A dotted group argument: an absent or null step reaches nothing; a supplied step of another shape is a problem
    for the whole call (its account and thread as its conversation), never read as a missing optional argument."""
    _declares(published_tool)
    name, descriptor = published_tool("write", touches=[
        {"conversation": "envelope.to", "account": "box.account", "effect": "write"}])
    fact = effects.effect_of(name, ext_tool=descriptor)
    addresses, found = effects.touched_addresses(fact, args)
    assert addresses == () and (found.startswith(problem) if problem else found == ""), found
    addresses, found = effects.touched_addresses(fact, {"envelope": {"to": "p@x"}, "box": {"account": "M"}})
    assert found == "" and [(a.conversation, a.account) for a in addresses] == [("p@x", "M")]


def test_an_executed_call_keeps_the_declaration_it_ran_with(published_tool):
    """What a dispatched call ran with rides in its result as plain facts, never its handler: restored from the
    recorded row it reads as the bound descriptor did, whatever a later publication declares; a row without them,
    or a descriptor bound to no reviewed declaration, keeps none."""
    import json

    _declares(published_tool)
    name, raw = published_tool("message", target="room", conversation="room", description="d" * 4000,
                               touches=[{"conversation": "to", "effect": "write"}])
    raw["handler"] = object()
    bound = registry.get_tool_stamped(name)
    kept = effects.retained_declaration(bound)
    assert json.loads(json.dumps(kept)) == kept and "description" not in kept["tool"] and "handler" not in kept
    row = {"tool": name, "args": {"room": "C1", "to": "p@x"}, effects.RETAINED_DECLARATION: kept}
    before = effects.effect_of(name, ext_tool=effects.retained_descriptor(row), args=row["args"])
    published_tool.bundle.generation_digest = raw["extension_generation"] = "generation-two"
    published_tool("read", target=[])
    after = effects.effect_of(name, ext_tool=effects.retained_descriptor(row), args=row["args"])
    assert before == after and (after.effect, [g.conversation for g in after.touches]) == ("message", ["to"])
    assert effects.effect_of(name, args=row["args"]).effect == "read"  # today's publication answers only later calls
    assert effects.retained_descriptor({"tool": name}) is None and effects.retained_declaration(raw) is None


def test_beside_groups_a_declared_target_of_another_shape_is_never_absent(published_tool):
    """A declaration with groups reads its own target, conversation and thread as strictly as a group: a supplied
    value of another shape or an unreadable path refuses the call; absence still reaches nothing. A row without
    groups keeps its reading (no addresses, never a problem from here)."""
    _declares(published_tool)
    name, descriptor = published_tool("message", target="body.room", conversation="body.room", thread="topic",
                                      touches=[{"conversation": "to", "effect": "write"}])
    fact = effects.effect_of(name, ext_tool=descriptor)
    for args, problem in (({"body": "C1"}, "'body' holds a str, so 'body.room' cannot be read"),
                          ({"body": {"room": {"id": 1}}}, "'body.room' holds a dict"),
                          ({"body": {"room": "C1"}, "topic": [["t"]]}, "'topic' holds a list"),
                          ({"to": "p@x"}, ""), ({"body": None, "to": "p@x"}, "")):
        assert effects.touched_addresses(fact, args)[1].startswith(problem) if problem else (
            effects.touched_addresses(fact, args)[1] == ""), args
    legacy, plain = published_tool("message", target="body.room", conversation="body.room")
    assert effects.touched_addresses(effects.effect_of(legacy, ext_tool=plain), {"body": "C1"}) == ((), "")
