"""The voice contract (PluginAPI 2.1): provider catalogs, route facts, call configuration, registration."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from ouroboros import extension_plugin_api as plugin_api_impl
from ouroboros.contracts.live_conversation import CallTool, LiveConfig, LiveOptions
from ouroboros.contracts.realtime import (CatalogModel, CatalogVoice, ProviderCatalog, RealtimeRoute,
                                          RealtimeSetupError, RouteFacts)

REPO = pathlib.Path(__file__).resolve().parents[1]
ROUTE = RealtimeRoute("test_realtime", "wss://example.invalid/live", "test::")


def _model(model_id="m1", **kwargs):
    return CatalogModel(model_id, model_id.upper(), RouteFacts(conversation_window_tokens=90_000), **kwargs)


def test_route_facts_keep_the_reserve_inside_the_window_and_unknown_as_none():
    facts = RouteFacts(conversation_window_tokens=90_000, accepted_input_tokens=231_828, working_reserve_tokens=10_000)
    assert facts.conversation_window_tokens == 90_000
    assert RouteFacts().conversation_window_tokens is None  # unknown stays unknown
    with pytest.raises(RealtimeSetupError):
        RouteFacts(conversation_window_tokens=10_000, working_reserve_tokens=10_000)
    with pytest.raises(RealtimeSetupError):
        RouteFacts(conversation_window_tokens=0)
    with pytest.raises(RealtimeSetupError):
        RouteFacts(basis="o200k")  # no tokenizer in the runtime: provider or estimator units only


def test_catalog_model_pairs_thinking_levels_with_the_knob_and_notes_only_unhonoured_knobs():
    assert _model(thinking_levels=("low", "high"), honours=frozenset({"thinking"})).thinking_levels == ("low", "high")
    with pytest.raises(RealtimeSetupError):
        _model(thinking_levels=("low",))
    with pytest.raises(RealtimeSetupError):
        _model(honours=frozenset({"thinking"}))
    with pytest.raises(RealtimeSetupError):
        _model(honours=frozenset({"pause"}), notes=(("pause", "Not here."),))
    with pytest.raises(RealtimeSetupError):
        _model(honours=frozenset({"teleport"}))
    with pytest.raises(RealtimeSetupError):
        _model(voices=(CatalogVoice("eve"), CatalogVoice("eve")))
    assert _model(notes=(("pause", "The engine decides."),)).note("pause") == "The engine decides."


def test_provider_catalog_has_one_default_and_refuses_unknown_models():
    catalog = ProviderCatalog("Test", (_model("a", default=True), _model("b")), ("TEST_API_KEY",))
    assert catalog.model("").id == "a"
    assert catalog.model("b").id == "b"
    with pytest.raises(RealtimeSetupError) as refused:
        catalog.model("c")
    assert refused.value.code == "model_unavailable"
    with pytest.raises(RealtimeSetupError):
        ProviderCatalog("Test", (_model("a"), _model("b")), ("TEST_API_KEY",))
    with pytest.raises(RealtimeSetupError):
        ProviderCatalog("Test", (_model("a", default=True), _model("b", default=True)), ("TEST_API_KEY",))
    with pytest.raises(RealtimeSetupError):
        ProviderCatalog("Test", (_model("a", default=True),), ())


def test_live_config_has_no_cap_or_reserve_and_validates_choices():
    config = LiveConfig(enabled=True, provider="gemini_live:gemini", pause="long", interruptions=False)
    assert config.pause == "long" and not hasattr(config, "max_seconds") and not hasattr(config, "reservation_usd")
    with pytest.raises(RealtimeSetupError):
        LiveConfig(enabled=True)
    with pytest.raises(RealtimeSetupError):
        LiveConfig(pause="forever")
    with pytest.raises(RealtimeSetupError):
        LiveConfig(enabled="yes", provider="x:y")
    assert LiveOptions(languages=("ru-RU", "en-US")).languages == ("ru-RU", "en-US")
    with pytest.raises(RealtimeSetupError):
        LiveOptions(languages=["ru-RU"])
    with pytest.raises(RealtimeSetupError):
        LiveOptions(model_output_bound_tokens=0)


def _api(tmp_path, monkeypatch, *permissions):
    monkeypatch.delenv("OUROBOROS_EXTENSION_PROCESS_CHILD", raising=False)
    config = plugin_api_impl._PluginAPIConfig(skill_name="sk", permissions=permissions, env_allowlist=(),
                                              state_dir=tmp_path, settings_reader=lambda: {})
    return plugin_api_impl.PluginAPIImpl(config)


def test_provider_registration_requires_a_typed_catalog(tmp_path, monkeypatch):
    api = _api(tmp_path, monkeypatch, "realtime_provider")
    catalog = ProviderCatalog("Test", (_model("a", default=True),), ("TEST_API_KEY",))
    with pytest.raises(plugin_api_impl.ExtensionRegistrationError):
        api.register_realtime_provider("voice", route=ROUTE, session=lambda *a, **k: None, catalog={"models": []})
    api.register_realtime_provider("voice", route=ROUTE, session=lambda *a, **k: None, catalog=catalog)
    assert api._staged.realtime_providers["sk:voice"]["catalog"] is catalog
    assert "handler" not in api._staged.realtime_providers["sk:voice"]


def test_feature_registration_takes_async_call_tools_with_unique_names(tmp_path, monkeypatch):
    async def run(host):
        return None

    async def end(host, arguments):
        return ""

    api = _api(tmp_path, monkeypatch, "live_conversation")
    tool = CallTool("end_call", "End the call.", {"type": "object", "properties": {}}, end, silent=True)
    with pytest.raises(plugin_api_impl.ExtensionRegistrationError):
        api.register_live_conversation("call", run, configure=LiveConfig, call_tools=({"name": "end_call"},))
    with pytest.raises(plugin_api_impl.ExtensionRegistrationError):
        api.register_live_conversation("call", run, configure=LiveConfig, call_tools=(
            CallTool("end_call", "End.", {}, lambda host, arguments: ""),))
    with pytest.raises(plugin_api_impl.ExtensionRegistrationError):
        api.register_live_conversation("call", run, configure=LiveConfig, call_tools=(tool, tool))
    api.register_live_conversation("call", run, configure=LiveConfig, call_tools=(tool,))
    assert api._staged.conversation_features["sk:call"]["call_tools"] == (tool,)


def _plugin_catalog(skill):
    spec = importlib.util.spec_from_file_location(f"_catalog_{skill}", REPO / "skills" / skill / "plugin.py",
                                                  submodule_search_locations=[str(REPO / "skills" / skill)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CATALOG


@pytest.mark.parametrize("skill,window,credential", [
    ("gemini_live", 92_000, "GEMINI_API_KEY"),
    ("grok_realtime", 80_000, "XAI_API_KEY"),
    ("openai_realtime", 133_554, "OPENAI_API_KEY"),  # the measured acceptance of 20.10, not the documented 128,000
])
def test_built_in_engines_declare_labelled_dated_windows(skill, window, credential):
    catalog = _plugin_catalog(skill)
    default = catalog.model("")
    assert default.facts.conversation_window_tokens == window
    assert default.facts.source and default.facts.observed
    accepted = default.facts.accepted_input_tokens
    assert accepted is None or accepted >= window
    assert catalog.credential_settings == (credential,)
    assert catalog.privacy


def test_plugin_api_2_1_is_one_additive_generation_over_the_negotiable_2_0_surface():
    from ouroboros.contracts.plugin_api import (PLUGIN_API_SURFACE_FINGERPRINTS, PLUGIN_API_VERSION,
                                                VALID_EXTENSION_PERMISSIONS, plugin_api_surface_fingerprint)

    # The additive voice generation keeps the 2.0 surface negotiable; one generation, one fingerprint.
    assert PLUGIN_API_VERSION == "2.1"
    assert PLUGIN_API_SURFACE_FINGERPRINTS == {
        "2.0": "03fabdf4334e6b2bde217b4cb83a80faaebc773ddce870546dd2108b75de17ca",
        "2.1": plugin_api_surface_fingerprint(),
    }
    assert {"realtime_provider", "live_conversation", "call_peer"} <= VALID_EXTENSION_PERMISSIONS
