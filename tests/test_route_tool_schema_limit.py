"""A provider's physical ceiling on tool schemas per request (direct OpenAI: 128).

Main and the provider canary fit through the same function: under the ceiling
nothing changes; above it the overflow tail is left out, never a core, meta,
gateway or actor-loaded schema, and Main tells the actor which names it left out.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import loop as loop_module
from ouroboros import loop_model_call
from ouroboros.provider_models import tool_schema_limit
from ouroboros.tool_capabilities import CORE_TOOL_NAMES, MCP_GATEWAY_TOOL_NAME, META_TOOL_NAMES, ROUTE_TAIL_TOOL_NAMES
from ouroboros.tool_policy import PURPOSE_CHARS, fit_tool_schemas_to_limit, route_tool_limit_notice, tool_purpose
from tests.provider_contract_ci import (
    CANARY_TOOL_NAME,
    delegate_start_canary_arguments,
    full_registry_canary_tools,
    provider_canary_matrix,
    run_provider_contract_canary,
)

OPENAI_MAIN = "openai::gpt-5.6-terra"
PINNED = sorted(CORE_TOOL_NAMES | META_TOOL_NAMES | {MCP_GATEWAY_TOOL_NAME})  # the gateway keeps the list stable


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": name,
                                             "parameters": {"type": "object", "properties": {}}}}


def _catalog(total):
    """Extras FIRST, so a plain truncation would cut core schemas."""
    assert total > len(PINNED), "the pinned catalog alone is already this large"
    extras = [f"extra_{index:02d}" for index in range(total - len(PINNED))]
    return [_schema(name) for name in (*extras, *PINNED)]


def _names(schemas):
    return [schema["function"]["name"] for schema in schemas]


def test_the_ceiling_is_a_direct_openai_route_fact():
    assert tool_schema_limit(OPENAI_MAIN) == 128
    assert tool_schema_limit("openai/gpt-5.6-luna") is None  # OpenRouter accepted 129 in CI
    assert tool_schema_limit("anthropic::claude-sonnet-5") is None
    assert tool_schema_limit(OPENAI_MAIN, use_local=True) is None


def test_a_catalog_within_the_ceiling_is_unchanged():
    at_ceiling = _catalog(128)
    kept, left_out = fit_tool_schemas_to_limit(at_ceiling, 128)
    assert kept == at_ceiling and left_out == ()
    above = _catalog(129)
    assert fit_tool_schemas_to_limit(above, None) == (above, ())


def test_one_over_the_ceiling_leaves_out_the_last_unpinned_schema_only():
    catalog = _catalog(129)
    kept, left_out = fit_tool_schemas_to_limit(catalog, 128)
    last_extra = [name for name in _names(catalog) if name.startswith("extra_")][-1]
    assert left_out == (last_extra,)
    assert _names(kept) == [name for name in _names(catalog) if name != last_extra]  # order kept
    assert set(PINNED) <= set(_names(kept))
    kept, left_out = fit_tool_schemas_to_limit(catalog, 128, keep=(last_extra,))
    assert last_extra in _names(kept) and len(kept) == 128 and len(left_out) == 1


def _round_ctx(schemas, *, model=OPENAI_MAIN, use_local=False):
    return SimpleNamespace(
        tool_schemas=schemas, active_model=model, active_use_local=use_local, task_id="t-ceiling",
        messages=[{"role": "user", "content": "go"}],
        tools=SimpleNamespace(_ctx=SimpleNamespace(),
                              callable_rows=lambda namespace="": [{"name": name} for name in META_TOOL_NAMES]),
        defer_resource_wait=None, attempt_cap=None, accumulated_usage={}, active_context_mode="max",
    )


def test_main_fits_its_resident_list_in_place_and_names_what_it_left_out():
    resident = _catalog(129)
    ctx = _round_ctx(resident)
    loop_model_call._fit_route_tool_ceiling(ctx)
    assert ctx.tool_schemas is resident and len(resident) == 128
    left_out = next(name for name in _names(_catalog(129)) if name not in _names(resident))
    notice = ctx.messages[-1]["content"]
    assert "at most 128 tool schemas" in str(notice) and left_out in str(notice)
    assert "enable_tools" in str(notice)
    assert ctx.tools._ctx._route_left_out_tool_names == {left_out}

    # The actor loads it again (enable_tools appends): it stays, another one goes, and one new notice names it.
    resident.append(_schema(left_out))
    rows = len(ctx.messages)
    loop_model_call._fit_route_tool_ceiling(ctx)
    assert len(resident) == 128 and left_out in _names(resident)
    (second,) = ctx.tools._ctx._route_left_out_tool_names - {left_out}
    assert len(ctx.messages) == rows + 1
    assert f"Not loaded for now:\n- {second}: {second}\n" in ctx.messages[-1]["content"]
    assert "this task had 129 loaded" in ctx.messages[-1]["content"]
    loop_model_call._fit_route_tool_ceiling(ctx)  # within the ceiling now: no third notice
    assert len(ctx.messages) == rows + 1


def test_the_ceiling_keeps_a_tool_this_task_already_called():
    catalog = _catalog(129)
    called = [name for name in _names(catalog) if name.startswith("extra_")][-1]  # the tail a plain fit leaves out
    ctx = _round_ctx(list(catalog))
    ctx.messages.append({"role": "assistant", "tool_calls": [{"function": {"name": called}}]})
    loop_model_call._fit_route_tool_ceiling(ctx)
    assert called in _names(ctx.tool_schemas) and len(ctx.tool_schemas) == 128


def test_a_schema_the_actor_enabled_stays_though_no_fit_left_it_out_before(tmp_path, monkeypatch):
    """A cold continuation keeps only its fitted list and a late-registered tool was never left out:
    what enable_tools loads in this run is the actor's choice, so the ceiling keeps it."""
    from ouroboros import loop, provider_models
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    # The route tail would leave first; its own order is pinned by the role test below.
    catalog = [schema for schema in initial_tool_schemas(registry)
               if schema["function"]["name"] not in ROUTE_TAIL_TOOL_NAMES]
    unpinned = [name for name in _names(catalog) if name not in PINNED]
    chosen, tail = unpinned[-1], unpinned[-2]
    resident = [schema for schema in catalog if schema["function"]["name"] != chosen]
    monkeypatch.setitem(provider_models.PROVIDER_TOOL_SCHEMA_LIMITS, "openai", len(resident))  # at the ceiling
    loop._setup_dynamic_tools(registry, resident, [])
    assert "Call schemas loaded" in registry.execute("enable_tools", {"tools": chosen})
    ctx = _round_ctx(resident)
    ctx.tools = registry
    loop_model_call._fit_route_tool_ceiling(ctx)
    assert chosen in _names(resident) and tail not in _names(resident)
    assert registry._ctx._route_left_out_tool_names == {tail}

    # The same list with that schema appended by the host, not loaded by the actor: the tail rule applies.
    hosted = _round_ctx([*[schema for schema in catalog if schema["function"]["name"] != chosen],
                         next(schema for schema in catalog if schema["function"]["name"] == chosen)])
    loop_model_call._fit_route_tool_ceiling(hosted)
    assert chosen not in _names(hosted.tool_schemas) and tail in _names(hosted.tool_schemas)


def _installed_catalog(tmp_path, skill_tools):
    """The shipped registry's round-one schemas plus one default-on skill's tools, in the product's order."""
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    skill = [{"type": "function", "function": {"name": f"ext_17_unix_computer_use_tool_{index:02d}",
                                               "description": "One skill tool. Its details.",
                                               "parameters": {"type": "object", "properties": {}}}}
             for index in range(skill_tools)]
    return initial_tool_schemas(ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)) + skill


def test_a_route_that_cannot_carry_every_schema_sends_memory_and_work_and_names_the_tail(tmp_path):
    """The rule beside CORE_TOOL_NAMES: memory and the work in front of the mind are always sent; owner
    administration, rescue and one-off media wait behind every other schema; each left-out tool is named
    with its own purpose. A tail tool this task already called keeps its place, and one whole skill
    loaded on purpose still fits."""
    memory = {"journal_read", "journal_write", "workpad_read", "workpad_write", "memory_map",
              "memory_update_registry", "compact_context"}
    from ouroboros.tools.registry import ToolRegistry

    registered = set(ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)._entries)
    assert ROUTE_TAIL_TOOL_NAMES <= registered and ROUTE_TAIL_TOOL_NAMES.isdisjoint(CORE_TOOL_NAMES)
    # Installation upkeep: its reader takes ordinary order, its two actions wait in the tail; the skill runner is sent.
    assert "inspect_harness" not in CORE_TOOL_NAMES | ROUTE_TAIL_TOOL_NAMES and "skill_exec" in CORE_TOOL_NAMES
    assert {"maintain_harness", "free_storage"} <= ROUTE_TAIL_TOOL_NAMES
    for skill_tools in (25, 4):  # macOS/Linux and Windows default-on computer-use skills
        catalog = _installed_catalog(tmp_path, skill_tools)
        kept, left_out = fit_tool_schemas_to_limit(catalog, 128)
        assert len(kept) == 128 and memory <= set(_names(kept)) and memory <= CORE_TOOL_NAMES
        tail = [name for name in _names(catalog) if name in ROUTE_TAIL_TOOL_NAMES]
        assert tail and set(tail).isdisjoint(_names(kept))
        assert set(_names(kept)) - set(PINNED)  # ordinary schemas were sent while the whole tail waited
        called = tail[0]
        for chosen in ({"protect": (called,)}, {"first": (called,)}, {"keep": (called,)}):  # history, this load, an earlier one
            kept_called, _ = fit_tool_schemas_to_limit(catalog, 128, **chosen)
            assert called in _names(kept_called), chosen
        skill = [name for name in _names(catalog) if name.startswith("ext_")]
        loaded, _ = fit_tool_schemas_to_limit(catalog, 128, first=skill)
        assert set(skill) <= set(_names(loaded)) and memory <= set(_names(loaded))  # memory stays beside a skill
        notice = route_tool_limit_notice(OPENAI_MAIN, 128, len(catalog), left_out, selectors=META_TOOL_NAMES,
                                         schemas=catalog)
        by_name = {schema["function"]["name"]: schema for schema in catalog}
        for name in left_out:
            purpose, cut = tool_purpose(by_name[name])
            assert f"\n- {name}: {purpose}{'…' if cut else ''}\n" in notice
        assert notice == route_tool_limit_notice(OPENAI_MAIN, 128, len(catalog), left_out,
                                                 selectors=META_TOOL_NAMES, schemas=list(reversed(catalog)))
    # The one short purpose: the first sentence, bounded with or without a sentence stop.
    def described(text):
        return {"function": {"name": "a", "description": text}}

    assert tool_purpose(described("Does one thing. And more.")) == ("Does one thing.", False)
    assert tool_purpose(described("No stop here")) == ("No stop here", False)
    for text in ("word " * 60, "Long " * 60 + ". Then more."):
        purpose, cut = tool_purpose(described(text))
        assert len(purpose) == PURPOSE_CHARS and cut
    # Every other route carries the same catalog byte for byte, in its own order.
    for model, use_local in (("openai/gpt-5.6-luna", False), ("anthropic::claude-sonnet-5", False),
                             ("claudexor::codex=gpt-6.2-astra", False), (OPENAI_MAIN, True)):
        before = json.dumps(catalog, sort_keys=False)
        ctx = _round_ctx(json.loads(before), model=model, use_local=use_local)
        assert loop_model_call._fit_route_tool_ceiling(ctx) is False
        assert json.dumps(ctx.tool_schemas, sort_keys=False) == before and len(ctx.messages) == 1


def test_the_notice_names_an_mcp_tool_by_its_server_text_and_says_when_nothing_can_load_it():
    """An MCP schema carries the host frame; its purpose is the server's own first sentence, quoted as data.
    Without a selector the notice does not promise a load: it says so, and names a resident gateway."""
    from ouroboros.mcp_schema import _model_facing_description

    tool = SimpleNamespace(server_id="slack", description="Channel history. Reads one channel.",
                           metadata={"annotations": {"readOnlyHint": True}})
    framed = {"type": "function", "function": {"name": "mcp_slack__history", "parameters": {"type": "object"},
                                               "description": _model_facing_description(tool)}}
    bare = {"type": "function", "function": {"name": "mcp_slack__post", "parameters": {"type": "object"},
                                             "description": _model_facing_description(SimpleNamespace(
                                                 server_id="slack", description="", metadata={}))}}
    assert tool_purpose(framed) == ("Channel history.", False) and tool_purpose(bare) == ("", False)

    def server(description, metadata=None):
        return tool_purpose({"function": {"name": "mcp_inv__builds", "description": _model_facing_description(
            SimpleNamespace(server_id="inv", description=description, metadata=metadata or {}))}})[0]

    # Leading blank lines, and the host's own labels inside the server's text, keep the server's purpose.
    assert server("\n\n\nLists builds. More.") == "Lists builds."
    assert server("Inventory\n\nServer metadata: returns build information.") == (
        "Inventory Server metadata: returns build information.")
    assert server("Inventory\n\nServer description: builds.", {"version": "2"}) == "Inventory Server description: builds."
    # Server text nested deeper than the decoder walks is read as text, never raised out of the catalog.
    deep = "Lists builds.\n\nServer metadata: " + '{"a":' * 100_000 + "1" + "}" * 100_000
    assert server(deep).startswith("Lists builds.")
    schemas = [_schema("read_file"), framed, bare]
    left = ("mcp_slack__history", "mcp_slack__post")
    notice = route_tool_limit_notice(OPENAI_MAIN, 1, 3, left, selectors=META_TOOL_NAMES, schemas=schemas)
    assert "\n- mcp_slack__history: 'Channel history.'\n- mcp_slack__post\n" in notice
    assert "External MCP tool" not in notice and "They stay callable: list_available_tools" in notice
    # Without the loader nothing promises a load: a usable list still shows them.
    listed = route_tool_limit_notice(OPENAI_MAIN, 1, 3, left, selectors={"list_available_tools"}, schemas=schemas)
    assert "They stay callable" not in listed and listed.endswith(
        "This task has no schema loader, so their schemas cannot be loaded here "
        "(list_available_tools marks them not loaded).")
    blind = route_tool_limit_notice(OPENAI_MAIN, 1, 3, left, selectors=(), schemas=schemas)
    assert "They stay callable" not in blind and blind.endswith(
        "This task has no schema loader, so their schemas cannot be loaded here.")
    gated = route_tool_limit_notice(OPENAI_MAIN, 2, 4, left, selectors=(),
                                    schemas=[*schemas, _schema(MCP_GATEWAY_TOOL_NAME)])
    assert gated.endswith(f"; {MCP_GATEWAY_TOOL_NAME} reaches the MCP ones this task is permitted to call.")


def test_main_leaves_fitting_catalogs_and_other_routes_untouched():
    for schemas, model, use_local in ((_catalog(128), OPENAI_MAIN, False),
                                      (_catalog(129), "openai/gpt-5.6-luna", False),
                                      (_catalog(129), OPENAI_MAIN, True)):
        before = _names(schemas)
        ctx = _round_ctx(schemas, model=model, use_local=use_local)
        loop_model_call._fit_route_tool_ceiling(ctx)
        assert _names(ctx.tool_schemas) == before
        assert ctx.messages == [{"role": "user", "content": "go"}]
        assert not hasattr(ctx.tools._ctx, "_route_left_out_tool_names")


def test_the_main_round_measures_and_sends_the_fitted_list(monkeypatch):
    seen = {}
    monkeypatch.setattr(loop_model_call, "_append_routing_receipts", lambda ctx: False)
    monkeypatch.setattr(loop_model_call, "_project_wake_input", lambda ctx, **_kw: False)
    monkeypatch.setattr(loop_module, "_measure_round_main_fit",
                        lambda ctx, **_kw: seen.setdefault("measured", len(ctx.tool_schemas)) and None)

    def dispatch(ctx, disposition, **_kw):
        seen["sent"] = len(ctx.tool_schemas)
        return {"role": "assistant", "content": "ok"}, 0.0

    monkeypatch.setattr(loop_module, "_dispatch_round_model", dispatch)
    below = len(PINNED) + 1  # one extra under the ceiling: nothing is cut
    for total, expected in ((129, 128), (128, 128), (below, below)):
        seen.clear()
        loop_model_call._call_round_model(_round_ctx(_catalog(total)))
        assert seen == {"measured": expected, "sent": expected}


def test_the_canary_sends_what_main_would_send_on_each_route():
    from tests.test_provider_contract_ci import _canonical_canary_call, _fake_usage

    catalog = full_registry_canary_tools()
    for canary_id in ("openai_direct_light", "openrouter_gpt"):  # single-turn rows
        canary = next(row for row in provider_canary_matrix() if row.canary_id == canary_id)
        sent = []

        def chat(canary=canary, sent=sent, **kwargs):
            sent.append(_names(kwargs["tools"]))
            call = _canonical_canary_call("call-ceiling", delegate_start_canary_arguments("ceiling"))
            return {"role": "assistant", "content": "", "tool_calls": [call]}, _fake_usage(canary, 1)

        run_provider_contract_canary(SimpleNamespace(chat=chat), canary=canary, tools=catalog, nonce="ceiling")
        limit = tool_schema_limit(canary.model)
        expected, _left_out = fit_tool_schemas_to_limit(catalog, limit, keep=(CANARY_TOOL_NAME,))
        assert sent == [_names(expected)] and CANARY_TOOL_NAME in sent[0]
        assert len(sent[0]) == (min(len(catalog), limit) if limit else len(catalog))
