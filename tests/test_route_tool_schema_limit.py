"""A provider's physical ceiling on tool schemas per request (direct OpenAI: 128).

Main and the provider canary fit through the same function: under the ceiling
nothing changes; above it the overflow tail is left out, never a core, meta or
actor-loaded schema, and Main tells the actor which names it left out.
"""

from __future__ import annotations

from types import SimpleNamespace

from ouroboros import loop as loop_module
from ouroboros import loop_model_call
from ouroboros.provider_models import tool_schema_limit
from ouroboros.tool_capabilities import CORE_TOOL_NAMES, META_TOOL_NAMES
from ouroboros.tool_policy import fit_tool_schemas_to_limit
from tests.provider_contract_ci import (
    CANARY_TOOL_NAME,
    delegate_start_canary_arguments,
    full_registry_canary_tools,
    provider_canary_matrix,
    run_provider_contract_canary,
)

OPENAI_MAIN = "openai::gpt-5.6-terra"
PINNED = sorted(CORE_TOOL_NAMES | META_TOOL_NAMES)


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": name,
                                             "parameters": {"type": "object", "properties": {}}}}


def _catalog(total):
    """Extras FIRST, so a plain truncation would cut core schemas."""
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
        messages=[{"role": "user", "content": "go"}], tools=SimpleNamespace(_ctx=SimpleNamespace()),
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
    assert len(ctx.messages) == rows + 1 and f"Not loaded for now: {second}." in ctx.messages[-1]["content"]
    assert "this task had 129 loaded" in ctx.messages[-1]["content"]
    loop_model_call._fit_route_tool_ceiling(ctx)  # within the ceiling now: no third notice
    assert len(ctx.messages) == rows + 1


def test_a_schema_the_actor_enabled_stays_though_no_fit_left_it_out_before(tmp_path, monkeypatch):
    """A cold continuation keeps only its fitted list and a late-registered tool was never left out:
    what enable_tools loads in this run is the actor's choice, so the ceiling keeps it."""
    from ouroboros import loop, provider_models
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    catalog = initial_tool_schemas(registry)
    unpinned = [name for name in _names(catalog) if name not in PINNED]
    chosen, tail = unpinned[-1], unpinned[-2]
    resident = [schema for schema in catalog if schema["function"]["name"] != chosen]
    monkeypatch.setitem(provider_models.PROVIDER_TOOL_SCHEMA_LIMITS, "openai", len(resident))  # at the ceiling
    loop._setup_dynamic_tools(registry, resident, [])
    assert "registered late" in registry.execute("enable_tools", {"tools": chosen})
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
    for total, expected in ((129, 128), (128, 128), (100, 100)):
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
