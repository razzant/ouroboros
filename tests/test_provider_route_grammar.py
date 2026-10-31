"""The kind table (C1) and the one route grammar (C3).

A route string names its connection: ``kind@name::model``. The tables every
credential consumer reads derive from ``PROVIDER_KINDS``; ``parse_model_route`` is
the only parser, ``provider_for_model`` its kind, and ``normalize_model_identity``
drops a connection name except for OpenAI-compatible servers.
"""

from __future__ import annotations

import pytest

from ouroboros import provider_models as pm
from ouroboros.provider_models import RouteParts, normalize_model_identity, parse_model_route, provider_for_model


@pytest.mark.parametrize("value,expected", [
    ("anthropic/claude-opus-5.5", RouteParts("openrouter", "", "anthropic/claude-opus-5.5")),
    ("openrouter::x-ai/grok", RouteParts("openrouter", "", "x-ai/grok")),
    ("openrouter@byok::anthropic/claude", RouteParts("openrouter", "byok", "anthropic/claude")),
    ("anthropic@default::claude", RouteParts("anthropic", "", "claude")),
    ("openai-compatible@vllm-2::llama", RouteParts("openai-compatible", "vllm-2", "llama")),
    ("claudexor::codex=gpt", RouteParts("claudexor", "", "codex=gpt")),
    ("claudexor@acct::codex=gpt", RouteParts("claudexor", "acct", "codex=gpt", valid=False)),
    ("openrouter@Bad Name::x", RouteParts("openrouter", "Bad Name", "x", valid=False)),
    ("openrouter@::x", RouteParts("openrouter", "", "x", valid=False)),
    ("openai::gpt (local)", RouteParts("local", "", "openai::gpt", local=True)),
    ("foo@bar::x", RouteParts("openrouter", "", "foo@bar::x")),  # an unknown head stays an opaque id
])
def test_parse_model_route(value, expected):
    assert parse_model_route(value) == expected


def test_provider_for_model_is_the_parsed_kind():
    assert provider_for_model("openrouter@byok::anthropic/claude") == "openrouter"
    assert provider_for_model("anthropic@k2::claude-opus-5-5") == "anthropic"
    assert provider_for_model("x (local)") == "local"
    assert provider_for_model("vendor/model") == "openrouter"
    assert provider_for_model("claudexor::codex=gpt") == "claudexor"


@pytest.mark.parametrize("value,expected", [
    # Built-in spellings keep the identities they always had.
    ("anthropic/claude-opus-4.6", "anthropic/claude-opus-4-6"),
    ("openai::gpt-5.6", "openai/gpt-5.6"),
    ("anthropic::claude-opus-5.5", "anthropic/claude-opus-5-5"),
    ("openai-compatible::llama", "openai-compatible/llama"),
    ("zai::glm-5.3 (local)", "zai/glm-5.3"),
    ("claudexor::codex=gpt", "claudexor::codex=gpt"),
    # A connection name says which key pays, not which model answers...
    ("openrouter@byok::anthropic/claude-opus-5.5", "anthropic/claude-opus-5-5"),
    ("anthropic@k2::claude-opus-5.5", "anthropic/claude-opus-5-5"),
    ("openrouter::z-ai/glm", "z-ai/glm"),
    # ...except on compatible servers, which never share learned facts.
    ("openai-compatible@vllm::llama", "openai-compatible@vllm/llama"),
    ("openrouter@Bad::x", "openrouter@Bad::x"),
])
def test_normalize_model_identity(value, expected):
    assert normalize_model_identity(value) == expected


def test_tables_derive_from_the_kind_table():
    assert list(pm.PROVIDER_KINDS) == ["openrouter", "openai", "anthropic", "deepseek", "zai", "minimax",
                                       "cloudru", "gigachat", "openai-compatible"]
    assert pm.PROVIDER_ENV_KEYS == {kind: spec.fields[0][1] for kind, spec in pm.PROVIDER_KINDS.items()
                                    if kind not in {"gigachat", "openai-compatible"}}
    secrets = {flat for spec in pm.PROVIDER_KINDS.values() for _field, flat, secret in spec.fields if secret}
    assert pm.MODEL_PROVIDER_CREDENTIAL_KEYS == secrets | {pm.CONNECTIONS_KEY}
    assert secrets == {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
                       "ZAI_API_KEY", "MINIMAX_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY",
                       "GIGACHAT_CREDENTIALS", "GIGACHAT_PASSWORD", "OPENAI_COMPATIBLE_API_KEY"}
    assert pm.PROVIDER_CREDENTIAL_GROUPS["openai-compatible"][-2:] == ("OPENAI_API_KEY", "OPENAI_BASE_URL")
    assert pm.ROUTE_CREDENTIAL_SETTING_KEYS == pm.ALL_PROVIDER_CREDENTIAL_KEYS | {pm.CONNECTIONS_KEY}
    assert {"OPENAI_API_KEY", "OPENAI_BASE_URL"} <= pm.ROUTE_CREDENTIAL_SETTING_KEYS
    assert {prefix for prefix, _kind in pm.PROVIDER_PREFIXES} == {f"{k}::" for k in ("claudexor", *pm.PROVIDER_KINDS)}


def test_readiness_keys_keep_the_review_migration_digest_membership():
    # review_pool_migration keys its receipts by this presence set: the membership is pinned.
    assert pm.READINESS_SETTING_KEYS == {
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
        "ZAI_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_USER",
        "GIGACHAT_PASSWORD", "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL",
    }


def test_connections_setting_is_registered_secret_and_not_exported():
    from ouroboros.config import SETTINGS_DEFAULTS, SETTINGS_KEYS_NOT_EXPORTED_TO_ENV, settings_env_keys
    from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS

    assert SETTINGS_DEFAULTS[pm.CONNECTIONS_KEY] == []
    assert pm.CONNECTIONS_KEY in SETTINGS_KEYS_NOT_EXPORTED_TO_ENV
    assert pm.CONNECTIONS_KEY not in settings_env_keys()
    assert pm.CONNECTIONS_KEY in FORBIDDEN_SKILL_SETTINGS


def test_an_environment_value_is_read_when_the_document_has_none(monkeypatch, tmp_path):
    import json
    from ouroboros import config

    rows = [{"kind": "openrouter", "name": "byok", "api_key": "env-key"}]
    saved = [{"kind": "openrouter", "name": "saved", "api_key": "doc-key"}]
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv(pm.CONNECTIONS_KEY, json.dumps(rows))
    assert config.load_settings()[pm.CONNECTIONS_KEY] == rows
    # An empty saved list is no value, like an empty string; saved rows win over the environment.
    (tmp_path / "settings.json").write_text(json.dumps({pm.CONNECTIONS_KEY: []}), encoding="utf-8")
    assert config.load_settings()[pm.CONNECTIONS_KEY] == rows
    (tmp_path / "settings.json").write_text(json.dumps({pm.CONNECTIONS_KEY: saved}), encoding="utf-8")
    assert config.load_settings()[pm.CONNECTIONS_KEY] == saved
