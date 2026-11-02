"""Named API connections (C2, C4): one resolver reads the connection a route names.

A named row carries its kind's fields and reaches the SAME per-kind branch as the
built-in connection; an unknown, disabled, mismatched or misspelled one refuses
before dispatch and is never served by another connection's key. Built-in routes
keep their capability/window fingerprints byte for byte. Keys are fixtures.
"""

from __future__ import annotations

import pytest

from ouroboros.llm_attempt import ConnectionUnavailable, ProviderPolicyRefusal
from ouroboros.llm_routing import OPENROUTER_METADATA_HEADERS, resolve_target
from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS
from ouroboros.provider_connections import (
    configured_connections, connection_references, connections_meta, normalize_connections,
)
from ouroboros.provider_models import (
    CONNECTIONS_KEY, DEEPSEEK_BASE_URL, MINIMAX_REGION_ENDPOINTS, ZAI_PLAN_ENDPOINTS, capability_base_url,
    compatible_only_main_model, exclusive_direct_kind, model_has_credentials, model_has_credentials_in_settings,
    provider_credential_plan,
)

ROWS = [
    {"kind": "openrouter", "name": "byok", "label": "Claude credits", "api_key": "row-or", "provider_routing": "repro"},
    {"kind": "openai", "name": "work", "api_key": "row-oa"},
    {"kind": "anthropic", "name": "k2", "api_key": "row-an"},
    {"kind": "deepseek", "name": "ds2", "api_key": "row-ds"},
    {"kind": "zai", "name": "coding", "api_key": "row-z", "plan": "coding"},
    {"kind": "minimax", "name": "cn", "api_key": "row-mm", "region": "cn_zh"},
    {"kind": "cloudru", "name": "cr2", "api_key": "row-cr", "base_url": "https://cloudru-2.example/v1"},
    {"kind": "gigachat", "name": "corp", "user": "u", "password": "row-pw", "base_url": "https://giga-2.example/v1",
     "scope": "GIGACHAT_API_CORP", "verify_ssl_certs": False},
    {"kind": "openai-compatible", "name": "vllm", "api_key": "row-ck", "base_url": "https://vllm.example/v1"},
    {"kind": "openai-compatible", "name": "ollama", "base_url": "http://127.0.0.1:11434/v1"},
]
BUILTIN = {
    "OPENROUTER_API_KEY": "or", "OPENAI_API_KEY": "oa", "ANTHROPIC_API_KEY": "an", "DEEPSEEK_API_KEY": "ds",
    "ZAI_API_KEY": "z", "MINIMAX_API_KEY": "mm", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr",
    "GIGACHAT_CREDENTIALS": "gc", "OPENAI_COMPATIBLE_API_KEY": "ck",
    "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
}


def _settings(rows=ROWS, **extra):
    return {**BUILTIN, CONNECTIONS_KEY: [dict(row) for row in rows], **extra}


@pytest.mark.parametrize("route,expected", [
    ("openrouter@byok::anthropic/claude-opus-5.5", {
        "provider": "openrouter", "api_key": "row-or", "base_url": "https://openrouter.ai/api/v1",
        "provider_routing": "repro", "usage_model": "openrouter@byok::anthropic/claude-opus-5.5",
        "default_headers": {**OPENROUTER_APP_HEADERS, **OPENROUTER_METADATA_HEADERS},
        "supports_openrouter_extensions": True}),
    ("openai@work::gpt-5.6-sol", {"provider": "openai", "api_key": "row-oa", "base_url": "https://api.openai.com/v1"}),
    ("anthropic@k2::claude-opus-5.5", {"provider": "anthropic", "api_key": "row-an", "resolved_model": "claude-opus-5-5",
                                       "usage_model": "anthropic@k2::claude-opus-5-5"}),
    ("deepseek@ds2::deepseek-v4-pro", {"provider": "deepseek", "api_key": "row-ds", "base_url": DEEPSEEK_BASE_URL,
                                       "requires_reasoning_echo": True}),
    ("zai@coding::glm-5.3", {"provider": "zai", "api_key": "row-z", "base_url": ZAI_PLAN_ENDPOINTS["coding"]}),
    ("minimax@cn::MiniMax-M3", {"provider": "minimax", "api_key": "row-mm", "base_url": MINIMAX_REGION_ENDPOINTS["cn_zh"],
                                "reasoning_split": True}),
    ("cloudru@cr2::zai-org/GLM-4.7", {"provider": "cloudru", "api_key": "row-cr",
                                       "base_url": "https://cloudru-2.example/v1"}),
    ("gigachat@corp::GigaChat-2-Max", {"provider": "gigachat", "api_key": "", "user": "u", "password": "row-pw",
                                        "base_url": "https://giga-2.example/v1", "scope": "GIGACHAT_API_CORP",
                                        "verify_ssl_certs": False}),
    ("openai-compatible@vllm::llama", {"provider": "openai-compatible", "api_key": "row-ck",
                                       "base_url": "https://vllm.example/v1", "keyless": False}),
    ("openai-compatible@ollama::llama", {"provider": "openai-compatible", "api_key": "",
                                         "base_url": "http://127.0.0.1:11434/v1", "keyless": True}),
])
def test_a_named_connection_reads_its_row_through_the_kind_branch(route, expected):
    target = resolve_target(route, _settings())
    assert {key: target[key] for key in expected} == expected
    assert target["connection"] == route.partition("@")[2].partition("::")[0]
    assert target["api_key"] not in set(BUILTIN.values())  # never the built-in key


@pytest.mark.parametrize("route,key,base_url", [
    ("anthropic/claude-opus-5.5", "or", "https://openrouter.ai/api/v1"),
    ("openai::gpt-5.6-sol", "oa", "https://api.openai.com/v1"),
    ("anthropic@default::claude-opus-5-5", "an", "https://api.anthropic.com/v1"),
    ("openai-compatible::llama", "ck", "https://compat.example/v1"),
])
def test_the_built_in_connection_reads_the_flat_keys(route, key, base_url):
    target = resolve_target(route, _settings())
    assert (target["api_key"], target["base_url"], target["connection"]) == (key, base_url, "")
    if target["provider"] == "openrouter":
        assert target["provider_routing"] is None
        assert target["default_headers"]["X-OpenRouter-Metadata"] == "enabled"


def test_the_ambient_resolver_reads_the_runtime_view(monkeypatch):
    import json

    monkeypatch.setenv("OPENROUTER_API_KEY", "env-or")
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps(ROWS))  # an operator's variable; WP3 adds the live document view
    assert resolve_target("openrouter@byok::x/y")["api_key"] == "row-or"
    assert resolve_target("x/y")["api_key"] == "env-or"


@pytest.mark.parametrize("route,rows,reason", [
    ("openrouter@gone::x/y", ROWS, "missing"),
    ("openrouter@empty::x/y", [{"kind": "openrouter", "name": "empty"}], "missing"),
    ("openai-compatible@nourl::m", [{"kind": "openai-compatible", "name": "nourl", "api_key": "k"}], "missing"),
    ("openrouter@byok::x/y", [{**ROWS[0], "enabled": False}], "disabled"),
    ("anthropic@byok::claude", ROWS, "kind_mismatch"),
    ("openrouter@Bad::x/y", ROWS, "invalid"),
    ("claudexor@acct::codex=gpt", ROWS, "invalid"),
    ("x/y", [{"kind": "openrouter", "name": "default", "enabled": False}], "disabled"),
    ("openai-compatible::m", [{"kind": "openai-compatible", "name": "default", "enabled": False}], "disabled"),
])
def test_an_unavailable_connection_refuses_and_never_falls_through(route, rows, reason):
    with pytest.raises(ConnectionUnavailable) as caught:
        resolve_target(route, _settings(rows))
    assert isinstance(caught.value, ProviderPolicyRefusal)
    assert (caught.value.code, caught.value.reason) == ("connection_unavailable", reason)
    # The twin: the same settings still serve every other connection.
    assert resolve_target("openai::gpt", _settings(rows))["api_key"] == "oa"


def test_the_refusal_is_classified_as_a_policy_refusal_not_a_provider_failure():
    from ouroboros.llm_attempt import PROVIDER_POLICY_REFUSAL
    from ouroboros.loop_llm_call import classify_llm_exception

    classification = classify_llm_exception(ConnectionUnavailable("openrouter@byok", "disabled"))
    assert classification.kind == PROVIDER_POLICY_REFUSAL
    assert classification.retry_same_request is False
    assert classification.provider_code == "connection_unavailable"


def test_presence_reads_the_named_connection():
    settings = _settings()
    assert model_has_credentials_in_settings("openrouter@byok::x/y", settings)
    assert not model_has_credentials_in_settings("openrouter@gone::x/y", settings)
    assert not model_has_credentials_in_settings("openrouter@byok::x/y", _settings([{**ROWS[0], "enabled": False}]))
    assert model_has_credentials_in_settings("openai-compatible@ollama::m", settings)  # a keyless endpoint
    off = _settings([{"kind": "openrouter", "name": "default", "enabled": False}])
    assert not model_has_credentials_in_settings("x/y", off) and model_has_credentials_in_settings("openai::g", off)


def test_ambient_presence_reads_the_runtime_view(monkeypatch):
    import json

    for key in BUILTIN:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(CONNECTIONS_KEY, json.dumps(ROWS[:1]))
    assert model_has_credentials("openrouter@byok::x/y") and not model_has_credentials("x/y")


def test_exclusivity_and_compatible_only_count_named_connections():
    only = {CONNECTIONS_KEY: [{"kind": "anthropic", "name": "k2", "api_key": "a"}]}
    assert exclusive_direct_kind(only) == "anthropic"
    assert exclusive_direct_kind({**only, CONNECTIONS_KEY: only[CONNECTIONS_KEY] + [ROWS[0]]}) == ""
    assert exclusive_direct_kind({"OPENAI_API_KEY": "k"}) == "openai"
    assert exclusive_direct_kind({"OPENAI_API_KEY": "k", "OPENAI_BASE_URL": "https://legacy/v1"}) == ""
    compat = {CONNECTIONS_KEY: [ROWS[-1]], "OUROBOROS_MODEL": "openai-compatible@ollama::m"}
    assert compatible_only_main_model(compat) == "openai-compatible@ollama::m"
    assert compatible_only_main_model({**compat, "OPENROUTER_API_KEY": "k"}) == ""


def test_configured_connections_are_in_owner_order():
    assert configured_connections(_settings())[:4] == [
        ("openrouter", ""), ("openrouter", "byok"), ("openai", ""), ("openai", "work")]
    assert ("openai-compatible", "ollama") in configured_connections(_settings())


def test_credential_plan_carries_the_rows_only_for_a_named_route():
    named = provider_credential_plan({"OUROBOROS_MODEL": "openrouter@byok::x/y"})
    assert CONNECTIONS_KEY in named["planned_keys"] and "OPENROUTER_API_KEY" in named["planned_keys"]
    assert CONNECTIONS_KEY not in provider_credential_plan({"OUROBOROS_MODEL": "x/y"})["planned_keys"]


def test_meta_lists_every_connection_as_the_picker_shows_it(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = _settings(ROWS + [{"kind": "deepseek", "name": "default", "enabled": False}])
    document = {key: value for key, value in settings.items() if key != "OPENROUTER_API_KEY"}
    meta = {row["id"]: row for row in connections_meta(settings, document)}
    assert list(meta)[:3] == ["openrouter", "openrouter@byok", "openai"]
    assert meta["openrouter"] == {"kind": "openrouter", "name": "", "label": "", "builtin": True, "enabled": True,
                                  "configured": True, "source": "environment", "id": "openrouter",
                                  "display": "OpenRouter"}
    assert meta["openrouter@byok"]["display"] == "OpenRouter · Claude credits"
    assert meta["openrouter@byok"]["source"] == "settings" and meta["openrouter@byok"]["builtin"] is False
    assert meta["deepseek"]["enabled"] is False and meta["deepseek"]["configured"] is True
    assert meta["gigachat@corp"]["display"] == "GigaChat · corp"
    assert meta["openai-compatible@ollama"]["configured"] is True
    unset = connections_meta({}, {})
    assert {row["source"] for row in unset} == {"none"} and len(unset) == 9


def test_references_name_rows_whose_connection_cannot_serve():
    settings = _settings([ROWS[0]], OUROBOROS_MODEL="openrouter@gone::x/y",
                         OUROBOROS_MODEL_FALLBACKS="openrouter@byok::x/y, anthropic@byok::claude",
                         OUROBOROS_SUBAGENTS={"enabled": True, "items": [
                             {"subagent_id": "scout", "route": {"kind": "api_model", "target_id": "openai@old::gpt"}}]})
    refs = connection_references(settings)
    assert refs == [
        {"setting": "OUROBOROS_MODEL", "route": "openrouter@gone::x/y", "connection": "openrouter@gone",
         "reason": "missing"},
        {"setting": "OUROBOROS_MODEL_FALLBACKS", "entry": "2", "route": "anthropic@byok::claude",
         "connection": "anthropic@byok", "reason": "kind_mismatch"},
        {"setting": "OUROBOROS_SUBAGENTS", "entry": "scout", "route": "openai@old::gpt", "connection": "openai@old",
         "reason": "missing"},
    ]
    assert connection_references(_settings(OUROBOROS_MODEL="openrouter@byok::x/y")) == []
    # The Safety Supervisor names a route like any role (MODEL_ROLE_SETTINGS), so a stale key there is reported too.
    assert connection_references(_settings([ROWS[0]], OUROBOROS_MODEL_SAFETY="openrouter@gone::x/y")) == [
        {"setting": "OUROBOROS_MODEL_SAFETY", "route": "openrouter@gone::x/y", "connection": "openrouter@gone",
         "reason": "missing"}]


@pytest.mark.parametrize("raw,fragment", [
    ("{}", "must be a list"),
    ("[not json", "JSON list"),
    ([{"kind": "openrouter", "name": ""}], "name must be"),
    ([{"kind": "bedrock", "name": "x"}], "unknown kind"),
    ([{"kind": "openrouter", "name": "Bad Name"}], "name must be"),
    ([ROWS[0], dict(ROWS[0])], "defined twice"),
    ([{"kind": "openrouter", "name": "default", "enabled": False, "api_key": "k"}], "carries only kind, name and enabled"),
    ([{"kind": "openrouter", "name": "byok", "apikey": "k"}], "unknown field"),
    ([{"kind": "openai", "name": "work", "provider_routing": "repro"}], "unknown field"),
    ([{"kind": "minimax", "name": "m", "region": "mars"}], "region must be global_en or cn_zh"),
    ([{"kind": "openrouter", "name": "byok", "provider_routing": "fastest"}], "provider_routing must be"),
    ([{"kind": "openrouter", "name": "byok", "enabled": "yes"}], "enabled must be"),
    ([{"kind": "openrouter", "name": "byok", "api_key": 7}], "api_key must be text"),
])
def test_save_validation_refuses_a_malformed_row(raw, fragment):
    rows, error = normalize_connections(raw)
    assert fragment in error and rows == []


def test_save_validation_accepts_valid_rows():
    raw = ROWS + [{"kind": "openrouter", "name": "default", "enabled": False},
                  {"kind": "openrouter", "name": "json", "provider_routing": '{"only": ["anthropic"]}', "api_key": "k"}]
    rows, error = normalize_connections(raw)
    assert error == "" and rows == raw


def test_masking_round_trip_keeps_secrets_by_row_identity():
    from ouroboros.gateway.secret_rows import ConnectionSecretIdentityAmbiguous, mask_connections, rehydrate_connections

    assert mask_connections(ROWS)[0]["api_key"] == "***"  # a short secret shows nothing of itself
    long_rows = [{**ROWS[0], "api_key": "sk-or-v1-0123456789"}, ROWS[7]]
    shown = mask_connections(long_rows)
    assert shown[0]["api_key"] == "sk-or-v1..." and shown[1]["password"] == "***set***"
    assert shown[0]["label"] == "Claude credits"  # only secrets are masked
    assert rehydrate_connections(shown, long_rows) == long_rows
    renamed = [{**shown[0], "name": "byok-2"}]
    assert rehydrate_connections(renamed, long_rows)[0]["api_key"] == ""  # a new identity gets no saved key
    with pytest.raises(ConnectionSecretIdentityAmbiguous):
        rehydrate_connections(shown[:1], [long_rows[0], dict(long_rows[0])])
    typed = [{**shown[0], "api_key": "sk-typed-new-key"}]
    assert rehydrate_connections(typed, long_rows)[0]["api_key"] == "sk-typed-new-key"


# --- Built-in fingerprints stay byte-identical (golden recorded on base 135965a2f) ----------------

_FP_SETTINGS = {
    "OPENROUTER_API_KEY": "or", "OPENAI_API_KEY": "oa", "OPENAI_BASE_URL": "https://legacy.example/v1",
    "OPENAI_COMPATIBLE_API_KEY": "ck", "OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1",
    "ANTHROPIC_API_KEY": "an", "DEEPSEEK_API_KEY": "ds", "ZAI_API_KEY": "z", "ZAI_PLAN": "coding",
    "MINIMAX_API_KEY": "mm", "MINIMAX_REGION": "cn_zh", "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr",
    "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://cloudru.example/v1", "GIGACHAT_CREDENTIALS": "gc",
    "GIGACHAT_BASE_URL": "https://giga.example/v1",
}
# model: (provider, window base_url, window fingerprint, response-limit fingerprint, observation scope, image scope)
# The image scope alone carries a key digest (#1520); OpenRouter's public catalog scope keeps its value.
_FP_GOLDEN = {
    "anthropic/claude-opus-5.5": ("openrouter", "", "f9face48876fd1cfaaed1fbb", "f9face48876fd1cfaaed1fbb",
                                  "880659b3d40f1ac984485bdb", "7495836c9654e2b823b949c9"),
    "anthropic::claude-opus-5-5": ("anthropic", "", "26f7e223f41e6e5c29406723", "26f7e223f41e6e5c29406723",
                                   "507f94d68a1ba37f3c4503b5", "32e01f936f89d9c57ef48d93"),
    "cloudru::zai-org/GLM-4.7": ("cloudru", "https://cloudru.example/v1", "e67857d121332e835b33969a",
                                 "e67857d121332e835b33969a", "cb57a7ccbd3559661d0b8f7f", "ca24bc8f99bbd5e697ca340e"),
    "deepseek::deepseek-v4-pro": ("deepseek", "", "06c8178656b2010301309144", "06c8178656b2010301309144",
                                  "f139a13d97f5a4b3ff047644", "59c2499f92bb95025681fe3c"),
    "gigachat::GigaChat-2-Max": ("gigachat", "https://giga.example/v1", "ed4fcddf4e7e7114fc2230ca",
                                 "ed4fcddf4e7e7114fc2230ca", "580638b7991b1b383d7b11ca", "609ef62dac8cf86ddb9072ff"),
    "minimax::MiniMax-M3": ("minimax", "https://api.minimaxi.com/v1", "84eecf31c2f34bf77918a567",
                            "84eecf31c2f34bf77918a567", "ce686cf9a23d0ef34098d339", "b7cbb3bdbb4d40e4573fc4ae"),
    "openai-compatible::llama-4": ("openai-compatible", "https://compat.example/v1", "59206c9bfcd807e33a1dfc9c",
                                   "59206c9bfcd807e33a1dfc9c", "78d9fed8f81e63eb5f7bb8aa", "f13e1125ecc0a9b594b5b459"),
    "openai::gpt-5.6-sol": ("openai", "https://legacy.example/v1", "6c0f0194c5d44fbb5a8aa070",
                            "32920fef79c6106d7f0e30fc", "cd38281dd7f0d4530a064ea6", "9e0d2b9d6c2d25a5fd42202d"),
    "openrouter::x-ai/grok-4.5": ("openrouter", "", "79986dee8ec98673e36a2502", "5183382b85782e5de61c6d0b",
                                  "6105dd529302b832a65426e0", "7495836c9654e2b823b949c9"),
    "zai::glm-5.3": ("zai", "https://api.z.ai/api/coding/paas/v4", "fdd75086685edf2a53a87426",
                     "fdd75086685edf2a53a87426", "3eb97e07521d4b5f5d6048d3", "8a607aa6a7e6b5528864fc8a"),
}


@pytest.mark.parametrize("model", sorted(_FP_GOLDEN))
def test_built_in_route_fingerprints_are_byte_identical(model, monkeypatch):
    from ouroboros.capability_evidence import route_fingerprint
    from ouroboros.context_window import route_observation_scope
    from ouroboros.gateway.settings import _active_main_route
    from ouroboros.response_limits import _response_fingerprint, dispatch_target
    from ouroboros.reviewer_window import reviewer_route
    from ouroboros.vision_routing import image_route_scope

    monkeypatch.setattr("ouroboros.config.runtime_settings", lambda: dict(_FP_SETTINGS))
    route = _active_main_route(_FP_SETTINGS, model_override=model)
    provider, base_url = route["provider"], route["base_url"]
    observed = (provider, base_url, route_fingerprint(provider=provider, base_url=base_url, model=model),
                _response_fingerprint(provider=provider, model=model, base_url=base_url),
                route_observation_scope(dict(route, options={}), _FP_SETTINGS),
                image_route_scope(dispatch_target(model, _FP_SETTINGS)))
    assert observed == _FP_GOLDEN[model]
    assert reviewer_route(model) == (provider, base_url)


def test_a_named_connection_fingerprints_its_own_endpoint_and_shares_a_fixed_one():
    settings = _settings()
    assert capability_base_url("openai-compatible@vllm::llama", settings) == "https://vllm.example/v1"
    assert capability_base_url("cloudru@cr2::m", settings) == "https://cloudru-2.example/v1"
    assert capability_base_url("openrouter@byok::x/y", settings) == capability_base_url("x/y", settings) == ""
    assert capability_base_url("openrouter@gone::x/y", settings) == ""
