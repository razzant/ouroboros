"""Connections in Nano: a server's guide behind its reader, handed over before its first use (K6).

The real registry, MCP manager, map producer and discovery tools over a stub transport
(``tests/_memory_scale_fixture.mcp_registry``: twelve servers whose own instructions are long, multi-line and quoted);
no server or model is called. Low and Max keep their maps byte for byte, for Main and for a helper.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client
from ouroboros.section_floor import HostSection
from ouroboros.tool_connections import connection_views, nano_connections_type
from ouroboros.tool_policy import initial_tool_schemas
from ouroboros.tools.tool_discovery import bind_resident_schemas, list_available_tools, resolve_gateway_call
from tests import _memory_scale_fixture as scale

# sha256 of the Low/Max maps of ``scale.mcp_registry``'s default shape, rendered by the code before this change
# (``ca500f776``): Main, a helper, and Main without either selector. Only a reviewed change of the Low/Max map renews them.
LOW_MAX = {
    "main": "0ce5d9ce2720fa8edd72f1c37a12fd2b608bfb9295eeb41bb7e64efb4cc14ec3",
    "helper": "074690c70289ff33ec5ccec5993c912b76490a36bf409cd4fa07be8ab6710322",
    "main without list_available_tools": "736e5f68b13c83ba613b8344b17c3d8b8ceee450699a758cbb201b9a59674bfa",
    "main without enable_tools": "832153e17d97d7c2e129fa3b8f3ba38ac4bf2030ff71aa52258e97aaf696bb94",
}


def _sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _kept_line(guide):
    """How a map that keeps a server's instructions prints them (``tool_connections._listed``)."""
    return "\n  Server instructions: " + " ".join(guide.split())


def _nano_actor(registry, *, mode="nano", start_tools=(), gateway=False):
    """The registry's context as a start in ``mode`` leaves it: the plan's captured maps, the active mode, the bound
    discovery tools over that mode's first tool list."""
    ctx = registry._ctx
    ctx.context_fit_plan = SimpleNamespace(core=SimpleNamespace(
        connections=connection_views(registry, helper=False, start_tools=start_tools)))
    ctx.active_context_mode = mode
    resident = initial_tool_schemas(registry, context_mode=mode, schema_names=start_tools or None)
    if gateway and not any(schema["function"]["name"] == "call_mcp_tool" for schema in resident):
        resident.append(registry.get_schema_by_name("call_mcp_tool"))
    bind_resident_schemas(registry, resident, gateway=gateway)
    return ctx, resident


def test_low_and_max_maps_are_unchanged_for_main_and_a_helper(tmp_path, monkeypatch):
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    main, _ = connection_views(registry, helper=False)
    helper, _ = connection_views(registry, helper=True)
    assert {"main": _sha(main), "helper": _sha(helper)} == {key: LOW_MAX[key] for key in ("main", "helper")}
    assert all(" ".join(guide.split()) in main and " ".join(guide.split()) in helper for guide in guides.values())
    for disabled in ("list_available_tools", "enable_tools"):
        registry._ctx.task_contract = {"disabled_tools": [disabled]}
        assert _sha(connection_views(registry, helper=False)[0]) == LOW_MAX[f"main without {disabled}"]


def test_the_nano_map_keeps_every_name_and_its_size_does_not_depend_on_instruction_length(tmp_path, monkeypatch):
    """Each server's name, tool count, residency and callable prefix, and every tool name, stay; its instructions
    leave Nano, so ten times longer instructions give the same bytes while the Low/Max map grows with them."""
    views = {}
    for factor in (1, 10):
        registry, guides = scale.mcp_registry(tmp_path / str(factor), monkeypatch, instruction_text=lambda i, f=factor: (
            f'"srv{i:02d}" rules:\n' + "\n".join(scale.prose(f"guide {i} {n}", 200) for n in range(4 * f))))
        views[factor] = connection_views(registry, helper=False), guides
    (low_max, nano), guides = views[1]
    (long_low_max, long_nano), long_guides = views[10]
    assert isinstance(nano, nano_connections_type()) and isinstance(nano, HostSection) and nano.section_id == "connections"
    assert str(nano) == str(long_nano) and len(long_low_max) > len(low_max)
    assert nano.guides == long_nano.guides == tuple(sorted(guides))
    assert nano.reader == 'list_available_tools(namespace="mcp_<server>")' and nano.operative_view is None
    for server in long_guides:
        assert scale.prose(f"guide {int(server[3:])} 0", 200) not in long_nano
        assert f"- mcp_{server} ({server.upper()}): " in long_nano
        assert f"on demand; callable as mcp_{server}__<name>; instructions at first use: tool_00, tool_01, " in long_nano
    listed = {row["name"] for row in mcp_client.get_manager().list_tools_for_registry()}
    assert len(listed) == scale.MCP_SHAPE["tools"]
    for name in listed:  # every MCP tool name is in the map (D3b: names never become counts)
        server, tool = name[4:].split("__", 1)
        line = next(line for line in long_nano.splitlines() if line.startswith(f"- mcp_{server} "))
        assert tool in line.rsplit(": ", 1)[1].split(", ")


def test_the_reader_returns_the_servers_own_text_exactly(tmp_path, monkeypatch):
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, resident = _nano_actor(registry)
    for server, guide in guides.items():
        assert "\n" in guide and '"' in guide  # line breaks and quotes, as servers write them
        listing = list_available_tools(registry, ctx, resident, namespace=f"mcp_{server}")
        assert f"\n{guide}\n[End of the server's instructions.]" in listing


def test_enable_tools_hands_the_guide_over_with_its_first_load_and_once(tmp_path, monkeypatch):
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    _nano_actor(registry)
    first = registry.execute("enable_tools", {"tools": "mcp_srv01__tool_00"})
    assert "Call schemas loaded" in first and guides["srv01"] in first
    assert all(guides[other] not in first for other in guides if other != "srv01")
    rest = registry.execute("enable_tools", {"tools": "mcp_srv01, mcp_srv02"})  # a second load of srv01's tools
    assert "Call schemas loaded" in rest and guides["srv01"] not in rest and guides["srv02"] in rest
    assert guides["srv02"] not in registry.execute("enable_tools", {"tools": "mcp_srv02"})  # already loaded


def test_a_view_that_keeps_the_guide_is_handed_nothing(tmp_path, monkeypatch):
    """The other direction: in Low/Max the map holds the instructions, so a load repeats none of them."""
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    _nano_actor(registry, mode="max")
    loaded = registry.execute("enable_tools", {"tools": "mcp_srv01"})
    assert "Call schemas loaded" in loaded and all(guide not in loaded for guide in guides.values())


def test_a_server_in_the_first_tool_list_keeps_its_guide_in_the_map(tmp_path, monkeypatch):
    """A start whose own tool list carries a server's schema (a call's or a parent's Nano base) can call it with no
    load before: that server's instructions stay in the map, and a later load hands nothing over again."""
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, resident = _nano_actor(registry, start_tools=("mcp_srv02__tool_00",))
    nano = ctx.context_fit_plan.core.connections[1]
    assert "mcp_srv02__tool_00" in {schema["function"]["name"] for schema in resident}
    assert _kept_line(guides["srv02"]) in nano and "srv02" not in nano.guides
    assert _kept_line(guides["srv03"]) not in nano and "srv03" in nano.guides
    assert guides["srv02"] not in registry.execute("enable_tools", {"tools": "mcp_srv02"})


@pytest.mark.parametrize("disabled", ["list_available_tools", "enable_tools"])
def test_an_actor_that_cannot_read_or_load_keeps_every_guide_in_its_map(tmp_path, monkeypatch, disabled):
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    registry._ctx.task_contract = {"disabled_tools": [disabled]}
    for helper in (False, True):
        nano = connection_views(registry, helper=helper)[1]
        assert nano.guides == () and nano.reader == "" and "instructions at first use" not in nano
        assert all(_kept_line(guide) in nano for guide in guides.values())


def _gateway_sends(monkeypatch):
    from ouroboros import safety
    from ouroboros.tools.tool_result import ToolResult

    sent = []
    monkeypatch.setattr(safety, "check_safety", lambda name, args, **_kw: (True, ""))
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", lambda name, args, _ctx=None: sent.append(
        (name, dict(args))) or ToolResult(status="ok", code="OK", text="sent"))
    return sent


def test_the_gateway_hands_the_guide_over_before_its_first_call(tmp_path, monkeypatch):
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, _ = _nano_actor(registry, gateway=True)
    sent = _gateway_sends(monkeypatch)
    envelope = {"id": "c", "type": "function", "function": {"name": "call_mcp_tool", "arguments": json.dumps(
        {"name": "mcp_srv03__tool_00", "arguments": {"id": "1"}})}}
    assert resolve_gateway_call(envelope, ctx) is envelope  # owed: the gateway's own handler runs
    assert resolve_gateway_call(envelope)["function"]["name"] == "mcp_srv03__tool_00"  # without a context: as before
    first = registry.execute_result("call_mcp_tool", {"name": "mcp_srv03__tool_00", "arguments": {"id": "1"}})
    assert sent == [] and guides["srv03"] in first.text and "Nothing was sent to mcp_srv03__tool_00" in first.text
    assert '"description": "An id"' in first.text  # the schema to compose the call from
    assert resolve_gateway_call(envelope, ctx)["function"]["name"] == "mcp_srv03__tool_00"
    second = registry.execute_result("call_mcp_tool", {"name": "mcp_srv03__tool_00", "arguments": {"id": "1"}})
    assert sent == [("mcp_srv03__tool_00", {"id": "1"})] and guides["srv03"] not in second.text
    shown = registry.execute("enable_tools", {"tools": "mcp_srv04"})  # the gateway shows schemas: the guide comes along
    assert "Call schemas for call_mcp_tool" in shown and guides["srv04"] in shown
    assert guides["srv04"] not in registry.execute_result(
        "call_mcp_tool", {"name": "mcp_srv04__tool_00", "arguments": {"id": "2"}}).text
    assert sent[-1] == ("mcp_srv04__tool_00", {"id": "2"})



def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _gw(call_id, tool, args):
    return _call(call_id, "call_mcp_tool", {"name": tool, "arguments": args})


class _Mind:
    """Responses through the executor (``loop_tool_execution.execute_tool_batch``) whose rows are recorded by the real
    recorder (``process_tool_results``) into one trace and transcript, as the loop and a call's tool lane record them;
    ``record=False`` is a batch cut short whose rows were never recorded."""

    def __init__(self, registry, tmp_path):
        self.registry, self.logs, self.trace, self.messages = registry, tmp_path / "logs", {"tool_calls": []}, []
        self.logs.mkdir(exist_ok=True)

    def respond(self, calls, *, record=True):
        from ouroboros.loop_tool_execution import execute_tool_batch, process_tool_results

        rows, control = execute_tool_batch(calls, self.registry, self.logs, "guide-hold", None, self.trace)
        assert control is None
        if record:
            process_tool_results(rows, self.messages, self.trace, lambda *_a, **_k: None, tools=self.registry)
        return [str(row["result"]) for row in rows]


def _responses(guide, texts):
    return sum(text.count(guide) for text in texts)


SHAPES = {  # one response's calls touching one owed server; the next response repeats them with new ids
    "two gateway calls": lambda n: [_gw(f"a{n}", "mcp_srv03__tool_00", {"id": "a"}), _gw(f"b{n}", "mcp_srv03__tool_01", {"id": "b"})],
    "enable then call": lambda n: [_call(f"e{n}", "enable_tools", {"tools": "mcp_srv03"}),
                                   _gw(f"g{n}", "mcp_srv03__tool_00", {"id": "a"})],
    "compact then call": lambda n: [_call(f"c{n}", "compact_context", {"working_note": "",
                                                                       "schema_names": ["mcp_srv03__tool_00"]}),
                                    _gw(f"g{n}", "mcp_srv03__tool_00", {"id": "a"})],
    "two direct calls": lambda n: [_call(f"d{n}", "mcp_srv03__tool_00", {"id": "a"}),
                                   _call(f"f{n}", "mcp_srv03__tool_01", {"id": "b"})],
}


@pytest.mark.parametrize("mode", ["nano", "low", "max"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_calls_of_an_owed_server_in_one_response_wait_for_the_next(tmp_path, monkeypatch, mode, shape):
    """K6's hand-over before the first send on every entry path (pr2rev1 G01, G02): in a Nano view no call of a server
    whose guide the mind has not read is sent in the response that touches it; the guide is returned once, a later use
    in the same response says it was returned earlier, and loads still load. The next response sends. In Low and Max,
    whose maps hold the guide, every call is sent at once and no guide is returned."""
    from ouroboros.tools.compact_context import record_context_view

    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, resident = _nano_actor(registry, mode=mode, gateway=True)
    record_context_view(ctx, [{"role": "user", "content": "hi"}], resident)
    sent = _gateway_sends(monkeypatch)
    mind = _Mind(registry, tmp_path)
    calls = [call for call in SHAPES[shape](1) if call["function"]["name"] != "enable_tools"
             and call["function"]["name"] != "compact_context"]
    first = mind.respond(SHAPES[shape](1))
    if mode != "nano":
        assert len(sent) == len(calls) and _responses(guides["srv03"], first) == 0
        return
    assert sent == [] and _responses(guides["srv03"], first) == 1 and guides["srv03"] in first[0]
    assert "returned earlier in this response" in first[1] and "Nothing was sent to mcp_srv03__tool_0" in first[1]
    if shape == "enable then call":
        assert "Call schemas for call_mcp_tool" in first[0]  # the load still loads (here: shows) its schemas
    second = mind.respond(SHAPES[shape](2))
    assert len(sent) == len(calls) and _responses(guides["srv03"], second) == 0


def test_a_direct_call_of_a_restored_schema_and_after_a_switch_into_nano_waits_for_its_guide(tmp_path, monkeypatch):
    """A schema the saved tool list restores (pr2rev1 G02) is called directly with no load in this process; a work
    that starts in Low and is rebound to Nano reads Nano's map from then on. Both hold the first call until the mind
    has read the guide; in Low the same call was sent at once."""
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, schemas = _nano_actor(registry, mode="low", gateway=True)
    bind_resident_schemas(registry, [*schemas, registry.get_schema_by_name("mcp_srv03__tool_00")], gateway=True)
    sent = _gateway_sends(monkeypatch)
    mind = _Mind(registry, tmp_path)
    mind.respond([_call("l1", "mcp_srv03__tool_00", {"id": "low"})])
    assert sent == [("mcp_srv03__tool_00", {"id": "low"})]
    ctx.active_context_mode = "nano"  # what bind_round_route leaves after a rebind into Nano
    (held,) = mind.respond([_call("n1", "mcp_srv03__tool_00", {"id": "nano"})])
    assert len(sent) == 1 and guides["srv03"] in held and "Call schema of mcp_srv03__tool_00" in held
    mind.respond([_call("n2", "mcp_srv03__tool_00", {"id": "nano"})])
    assert sent[-1] == ("mcp_srv03__tool_00", {"id": "nano"})


def test_a_held_resume_whose_saved_list_carries_a_server_holds_its_first_direct_call(tmp_path, monkeypatch):
    """pr2rev1 G03: the map of a held-note resume is captured with the caller's base, so a server the saved list makes
    resident stays behind the reader; its first direct call returns the guide and sends nothing."""
    from ouroboros import context, context_fit as fit, context_start_bound as bound

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
    registry, guides = scale.mcp_registry(tmp_path / "mcp", monkeypatch)
    ctx = registry._ctx
    env, memory = scale.env_on(scale.install(tmp_path / "drive"))
    saved = [*scale.tools(), registry.get_schema_by_name("mcp_srv02__tool_00")]
    ctx._held_note_capture = {"entries": [], "tool_schemas": saved}
    ctx.context_fit_plan = context.build_context_fit_plan(
        env, memory, {"id": "held-1", "type": "task", "chat_id": 1, "text": "resume", "_is_direct_chat": True},
        preferred_mode="nano", ctx=ctx, tool_schemas=scale.tools(), tool_registry=registry,
        route_resolver=scale.route_resolver(1_000_000, "provider", None))
    ctx.active_context_mode = "nano"
    assert "mcp_srv02__tool_00" in ctx.context_fit_plan.start_nano_base and "srv02" in ctx.context_fit_plan.core.connections[1].guides
    bind_resident_schemas(registry, saved, gateway=False)
    sent = _gateway_sends(monkeypatch)
    mind = _Mind(registry, tmp_path)
    (held,) = mind.respond([_call("r1", "mcp_srv02__tool_00", {"id": "1"})])
    assert sent == [] and guides["srv02"] in held
    mind.respond([_call("r2", "mcp_srv02__tool_00", {"id": "1"})])
    assert sent == [("mcp_srv02__tool_00", {"id": "1"})]


def test_an_unrecorded_response_still_owes_the_guide_and_a_restart_says_only_what_is_true(tmp_path, monkeypatch):
    """A batch cut short whose rows never reached the transcript (the loop raises its control before recording) leaves
    the guide owed: the next response returns it again rather than send. A fresh context after a restart (nothing is
    persisted) returns it again too, and the wording never claims a first use (pr2rev1 G04)."""
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, _ = _nano_actor(registry, gateway=True)
    sent = _gateway_sends(monkeypatch)
    mind = _Mind(registry, tmp_path)
    (cut,) = mind.respond([_gw("x1", "mcp_srv04__tool_00", {"id": "1"})], record=False)
    (again,) = mind.respond([_gw("x2", "mcp_srv04__tool_00", {"id": "1"})])
    assert sent == [] and guides["srv04"] in cut and guides["srv04"] in again
    mind.respond([_gw("x3", "mcp_srv04__tool_00", {"id": "1"})])
    assert sent == [("mcp_srv04__tool_00", {"id": "1"})]
    ctx._mcp_guides_handed, ctx._mcp_guides_pending = set(), {}  # a new ToolContext after a restore
    (restarted,) = mind.respond([_gw("x4", "mcp_srv04__tool_00", {"id": "4"})])
    assert len(sent) == 1 and guides["srv04"] in restarted
    assert "first use" not in restarted and restarted.startswith("Nothing was sent to mcp_srv04__tool_00")


def test_an_actor_whose_map_keeps_the_guides_sends_at_once_in_nano(tmp_path, monkeypatch):
    """The other direction at the executor's seam: an actor that cannot call the reader keeps every guide in its Nano
    map, so nothing is owed and its first direct and gateway calls are sent in the response that makes them."""
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    registry._ctx.task_contract = {"disabled_tools": ["list_available_tools"]}
    _nano_actor(registry, gateway=True)
    sent = _gateway_sends(monkeypatch)
    texts = _Mind(registry, tmp_path).respond([_call("d", "mcp_srv03__tool_00", {"id": "d"}),
                                              _gw("g", "mcp_srv05__tool_00", {"id": "g"})])
    assert sent == [("mcp_srv03__tool_00", {"id": "d"}), ("mcp_srv05__tool_00", {"id": "g"})]
    assert not any(guide in text for guide in guides.values() for text in texts)


@pytest.mark.parametrize("mode", ["nano", "low", "max"])
def test_schemas_the_mind_makes_resident_bring_their_servers_guide_once_and_only_in_nano(tmp_path, monkeypatch, mode):
    """``compact_context(schema_names=...)`` makes a server's tools resident without ``enable_tools``: in a Nano view the
    result that queues it carries that server's guide (before the next request, which is the first that can call the
    tool directly), once per server; a server already handed over and Low/Max (whose maps keep it) get none."""
    from ouroboros.tools.compact_context import _compact_context, record_context_view

    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    ctx, resident = _nano_actor(registry, mode=mode)
    record_context_view(ctx, [{"role": "user", "content": "hi"}], resident)
    first = _compact_context(ctx, working_note="", schema_names=["mcp_srv03__tool_00", "mcp_srv05__tool_01"])
    assert first.startswith("Working view requested.") and ctx._pending_compaction["schema_names"]
    assert all(guides[other] not in first for other in guides if other not in ("srv03", "srv05"))
    if mode != "nano":
        assert guides["srv03"] not in first and guides["srv05"] not in first
        return
    assert guides["srv03"] in first and guides["srv05"] in first
    again = _compact_context(ctx, working_note="", schema_names=["mcp_srv03__tool_00", "mcp_srv05__tool_01"])
    assert guides["srv03"] not in again and guides["srv05"] not in again
    assert guides["srv03"] not in registry.execute("enable_tools", {"tools": "mcp_srv03"})  # handed: not again


def test_the_ordinary_build_carries_the_nano_form_and_max_keeps_the_guides(tmp_path, monkeypatch):
    """One start's view before and after: through the real build the captured Nano map is the guide-less form and the
    Max view of the same capture holds every guide, so a plan that rebinds to Max mid-run shows them there."""
    from ouroboros import context, context_fit as fit, context_start_bound as bound

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    env, memory = scale.env_on(scale.install(tmp_path / "drive"))
    plan = context.build_context_fit_plan(env, memory, {"id": "start", "type": "task", "chat_id": 1, "text": "hi",
                                                        "_is_direct_chat": True}, preferred_mode="max",
                                          tool_schemas=scale.tools(), tool_registry=registry,
                                          route_resolver=scale.route_resolver(1_000_000, "provider", None))
    nano = plan.core.connections[1]
    assert isinstance(nano, nano_connections_type()) and isinstance(nano, HostSection) and nano.guides == tuple(sorted(guides))
    flat = {mode: json.dumps(plan.messages_for(mode), ensure_ascii=False) for mode in ("max", "nano")}
    for guide in guides.values():
        assert json.dumps(" ".join(guide.split()), ensure_ascii=False)[1:-1] in flat["max"]
        assert json.dumps(" ".join(guide.split()), ensure_ascii=False)[1:-1] not in flat["nano"]
    assert "mcp_srv11__<name>; instructions at first use: tool_00" in flat["nano"]


def test_a_start_base_that_carries_a_server_keeps_its_guide_through_the_build(tmp_path, monkeypatch):
    """The build reads the map with the start's own Nano base (``nano_base``, a call's or a parent's suggestion), so a
    server whose schema that base carries keeps its instructions, and the start bound prices the same map."""
    from ouroboros import context, context_fit as fit, context_start_bound as bound

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
    registry, guides = scale.mcp_registry(tmp_path, monkeypatch)
    env, memory = scale.env_on(scale.install(tmp_path / "drive"))
    plan = context.build_context_fit_plan(env, memory, {"id": "start", "type": "task", "chat_id": 1, "text": "hi",
                                                        "_is_direct_chat": True}, preferred_mode="nano",
                                          tool_schemas=scale.tools(), tool_registry=registry,
                                          nano_base=("list_available_tools", "enable_tools", "mcp_srv05__tool_00"),
                                          route_resolver=scale.route_resolver(1_000_000, "provider", None))
    nano = plan.core.connections[1]
    assert "srv05" not in nano.guides and _kept_line(guides["srv05"]) in nano
    assert "srv06" in nano.guides and _kept_line(guides["srv06"]) not in nano
