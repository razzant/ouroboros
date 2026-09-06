import json

import pytest

from ouroboros.server_runtime import (
    apply_runtime_provider_defaults,
    has_startup_ready_provider,
    needs_local_model_autostart,
)
from ouroboros.config import SETTINGS_DEFAULTS
from ouroboros.configured_subagents import resolve_configured_subagents


_NEWLY_RETIRED_SHIPPED_HEAVY_DEFAULTS = (
    "anthropic/claude-opus-4.7",
    "anthropic::claude-opus-4-7",
    "openai::gpt-5.6-sol",
    "anthropic::claude-opus-5",
    "cloudru::zai-org/GLM-4.7",
    "gigachat::GigaChat-3-Ultra",
    "gigachat::GigaChat-2-Max",
    "minimax::MiniMax-M3",
)
_LOCAL_HEAVY_VALUES_THAT_MUST_SURVIVE_GLOBAL_RETIREMENT = (
    "google/gemini-3.1-flash-lite",
    "openai::gpt-5.4",
    "openai/gpt-5.4-pro",
)


def test_has_startup_ready_provider_accepts_any_remote_key_or_local_routing():
    assert has_startup_ready_provider({"OPENROUTER_API_KEY": "sk-or-test"})
    assert has_startup_ready_provider({"OPENAI_API_KEY": "sk-openai"})
    assert has_startup_ready_provider({"ANTHROPIC_API_KEY": "sk-ant"})
    assert has_startup_ready_provider({"MINIMAX_API_KEY": "minimax-key"})
    assert has_startup_ready_provider({"OPENAI_COMPATIBLE_BASE_URL": "https://compat.example/v1"})
    assert not has_startup_ready_provider({"OPENAI_COMPATIBLE_API_KEY": "compat-key"})
    assert has_startup_ready_provider({"CLOUDRU_FOUNDATION_MODELS_API_KEY": "cloudru-key"})
    assert has_startup_ready_provider({"GIGACHAT_CREDENTIALS": "giga-creds"})
    assert has_startup_ready_provider({"USE_LOCAL_MAIN": True})
    assert has_startup_ready_provider({"USE_LOCAL_FALLBACK": "True"})
    assert not has_startup_ready_provider({"LOCAL_MODEL_SOURCE": "Qwen/Qwen2.5-7B-Instruct-GGUF"})


def test_consciousness_local_lane_autostarts_but_is_not_startup_ready():
    settings = {
        "LOCAL_MODEL_SOURCE": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "USE_LOCAL_CONSCIOUSNESS": True,
    }
    assert needs_local_model_autostart(settings)
    assert not has_startup_ready_provider(settings)


def test_enabled_local_api_subagents_autostart_without_becoming_root_readiness():
    from ouroboros.configured_subagents import SUBAGENTS_SETTING

    actor = json.dumps({
        "enabled": True,
        "items": [{
            "subagent_id": "local-scout",
            "name": "Local scout",
            "recommended_use": "Use the owner's local model.",
            "route": {"kind": "api_model", "target_id": "owner-model (local)"},
            "effort": "high",
        }],
    })
    actor_only = {SUBAGENTS_SETTING: actor}
    assert needs_local_model_autostart(actor_only)
    assert not has_startup_ready_provider(actor_only)
    assert needs_local_model_autostart({**actor_only, "OPENAI_API_KEY": "remote-main"})

    disabled = json.loads(actor)
    disabled["enabled"] = False
    assert not needs_local_model_autostart({SUBAGENTS_SETTING: json.dumps(disabled)})

    assert needs_local_model_autostart({
        "OUROBOROS_MODEL_HEAVY": "owner-heavy",
        "USE_LOCAL_HEAVY": True,
    })



_PROVIDER_ENV_KEYS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "MINIMAX_API_KEY", "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_BASE_URL", "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD",
)


def _read_time_review_models(monkeypatch, provider_env: dict) -> tuple[list, list]:
    """ABI 7.0 (ABI-10): the retired comma keys are never INTRODUCED into
    settings — the direct-provider review adaptation lives on the READ side
    (`get_review_models`/`get_scope_review_models` over the derived env plane).
    Returns (triad, scope) as that install class resolves them."""
    from ouroboros.config import get_review_models, get_scope_review_models

    for key in (*_PROVIDER_ENV_KEYS, "OUROBOROS_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
                "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT"):
        monkeypatch.delenv(key, raising=False)
    for key, value in provider_env.items():
        monkeypatch.setenv(key, value)
    return list(get_review_models() or []), list(get_scope_review_models() or [])


def test_apply_runtime_provider_defaults_autofills_official_openai_models():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "anthropic/claude-opus-4.6",
        "OUROBOROS_MODEL_HEAVY": "anthropic/claude-opus-4.6",
        "OUROBOROS_MODEL_LIGHT": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
    })

    assert changed
    assert set(changed_keys) == {
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_HEAVY",
        "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_FALLBACKS",
        # v6.82.0: deep self-review is a per-provider slot too, so a direct-only
        # install never keeps an unreachable OpenRouter-form id for it.
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    }
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.6-terra"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.6-luna"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai::gpt-5.6-sol"
    # ABI-10: retired comma keys are never introduced into settings.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized

    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "google/gemini-3.1-flash-lite",
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.1-flash-lite",
        "OUROBOROS_MODEL_LIGHT": "google/gemini-3.1-flash-lite",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_REVIEW_MODELS": (
            "openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.6"
        ),
    })

    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.6-terra"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.6-luna"
    assert normalized["OUROBOROS_REVIEW_MODELS"] == (
        "openai::gpt-5.6-terra,openai::gpt-5.6-terra,openai::gpt-5.6-terra"
    )
    # ABI-10: the scope keys were absent on input — never introduced.
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in normalized

    # Triad-expansion contract: the SETTINGS_DEFAULTS-fed case — the shipped
    # mixed-provider triad is foreign to an OpenAI-only install,
    # so it expands to the new OPENAI_DIRECT_DEFAULTS [main, light, light] fallback.
    payload = dict(SETTINGS_DEFAULTS)
    payload["OPENAI_API_KEY"] = "sk-openai"
    normalized, changed, changed_keys = apply_runtime_provider_defaults(payload)

    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.6-terra"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.6-luna"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai::gpt-5.6-sol"
    # ABI 7.0 (ABI-10): retired comma keys are never introduced into settings.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized


def test_openai_only_review_models_resolve_at_read_time(monkeypatch):
    triad, scope = _read_time_review_models(monkeypatch, {
        "OPENAI_API_KEY": "sk-openai", "OUROBOROS_MODEL": "openai::gpt-5.6-terra",
    })
    assert triad and all(m.startswith("openai::") for m in triad)
    assert scope and all(m.startswith("openai::") for m in scope)


def test_apply_runtime_provider_defaults_migrates_saved_openai_values():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "openai/gpt-5.5",
        "OUROBOROS_MODEL_HEAVY": "openai/gpt-5.5",
        "OUROBOROS_MODEL_LIGHT": "openai/gpt-4.1",
        "OUROBOROS_MODEL_FALLBACKS": "openai/gpt-4.1",
        "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5",
    })

    assert changed
    assert set(changed_keys) == {
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_HEAVY",
        "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_FALLBACKS",
        # v6.82.0: deep self-review is a per-provider slot too, so a direct-only
        # install never keeps an unreachable OpenRouter-form id for it.
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
        # ABI-10: only the comma key the payload actually CARRIED normalizes;
        # absent retired keys are never introduced.
        "OUROBOROS_REVIEW_MODELS",
    }
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in normalized
    # Active Main/Light/Fallback defaults migrate. The known shipped Heavy is
    # product-authored, not custom owner intent, so it does not become an actor.
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.6-terra"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.6-luna"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai::gpt-5.6-sol"
    # v6.36.0 (D4): an explicit provider-matching review list is honored EXACTLY
    # (1 model = 1 slot — a loud single_reviewer_no_diversity degraded mode), not
    # silently expanded to [main, light, light]. Expansion fires only when the
    # configured list is empty or contains foreign (non-provider) models.
    assert normalized["OUROBOROS_REVIEW_MODELS"] == "openai::gpt-5.5"


def test_apply_runtime_provider_defaults_clears_shipped_openai_heavy_and_keeps_review_models():
    # Sol was a product-authored Heavy default, while the review models below are
    # explicit owner choices and remain byte-for-byte unchanged.
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "openai::gpt-5.6-terra",
        "OUROBOROS_MODEL_HEAVY": "openai::gpt-5.6-sol",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai::gpt-5.6-sol",
        "OUROBOROS_REVIEW_MODELS": "openai::gpt-5.6-terra,openai::gpt-5.5-mini",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai::gpt-5.6-terra",  # already in direct format
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai::gpt-5.6-terra",  # already in direct format
    })

    assert changed
    assert changed_keys == ["OUROBOROS_MODEL_HEAVY"]
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_REVIEW_MODELS"] == "openai::gpt-5.6-terra,openai::gpt-5.5-mini"


def test_apply_runtime_provider_defaults_preserves_duplicate_scope_slots_for_openai():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "openai::gpt-5.6-sol",
        "OUROBOROS_MODEL_HEAVY": "openai::owner-selected-heavy",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai::gpt-5.6-sol",
        "OUROBOROS_REVIEW_MODELS": "openai::gpt-5.6-sol,openai::gpt-5.6-sol,openai::gpt-5.6-sol",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai::gpt-5.6-sol",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai::gpt-5.6-sol,openai::gpt-5.6-sol,openai::gpt-5.6-sol",
    })

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai::gpt-5.6-sol,openai::gpt-5.6-sol,openai::gpt-5.6-sol"


def test_apply_runtime_provider_defaults_clears_shipped_opus47_heavy_with_openrouter():
    current_openrouter = "anthropic/claude-opus-" + "4.7"
    current_claude_code = "claude-opus-" + "4-7[1m]"
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENROUTER_API_KEY": "sk-or",
        "OUROBOROS_MODEL": current_openrouter,
        "OUROBOROS_MODEL_HEAVY": current_openrouter,
        "OUROBOROS_REVIEW_MODELS": f"openai/gpt-5.5,{current_openrouter}",
        "CLAUDE_CODE_MODEL": current_claude_code,
    })

    assert changed
    assert changed_keys == ["OUROBOROS_MODEL_HEAVY"]
    assert normalized["OUROBOROS_MODEL"] == current_openrouter
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_REVIEW_MODELS"] == f"openai/gpt-5.5,{current_openrouter}"
    assert normalized["CLAUDE_CODE_MODEL"] == current_claude_code


@pytest.mark.parametrize("shipped_heavy", _NEWLY_RETIRED_SHIPPED_HEAVY_DEFAULTS)
def test_every_newly_retired_product_heavy_is_cleared_before_actor_migration(shipped_heavy):
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OUROBOROS_MODEL_HEAVY": shipped_heavy,
    })

    resolution = resolve_configured_subagents(normalized)
    rows = () if resolution.config is None else resolution.config.items
    assert changed
    assert changed_keys == ["OUROBOROS_MODEL_HEAVY"]
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert all(row.subagent_id != "legacy-heavy" for row in rows)


@pytest.mark.parametrize(
    "saved_heavy",
    (
        *_NEWLY_RETIRED_SHIPPED_HEAVY_DEFAULTS,
        *_LOCAL_HEAVY_VALUES_THAT_MUST_SURVIVE_GLOBAL_RETIREMENT,
    ),
)
def test_local_override_preserves_exact_heavy_value_as_explicit_local_actor(saved_heavy):
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OUROBOROS_MODEL_HEAVY": saved_heavy,
        "USE_LOCAL_HEAVY": True,
    })

    resolution = resolve_configured_subagents(normalized)
    assert not changed
    assert changed_keys == []
    assert resolution.config is not None
    assert [(row.subagent_id, row.route.target_id) for row in resolution.config.items] == [
        ("legacy-heavy", f"{saved_heavy} (local)"),
    ]


def test_apply_runtime_provider_defaults_refreshes_retired_gpt54_defaults():
    old_main = "openai/gpt-" + "5.4"
    old_pro = "openai/gpt-" + "5.4-pro"
    old_mini = "openai/gpt-" + "5.4-mini"
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENROUTER_API_KEY": "sk-or",
        "OUROBOROS_REVIEW_MODELS": f"{old_main},{old_mini}",
        "OUROBOROS_SCOPE_REVIEW_MODEL": old_pro,
        "OUROBOROS_SCOPE_REVIEW_MODELS": f"{old_pro},{old_mini}",
    })

    assert changed
    assert "OUROBOROS_REVIEW_MODELS" in changed_keys
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" in changed_keys
    # gpt-5.4 and gpt-5.4-pro are genuinely retired -> 5.5 / 5.5-pro. But gpt-5.4-mini
    # is a LIVE model (the 5.5 family has no mini lane), so it must pass through
    # unchanged rather than be rewritten to a non-existent gpt-5.5-mini.
    assert normalized["OUROBOROS_REVIEW_MODELS"] == "openai/gpt-5.5,openai/gpt-5.4-mini"
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "openai/gpt-5.5-pro"
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai/gpt-5.5-pro,openai/gpt-5.4-mini"


def test_apply_runtime_provider_defaults_migrates_legacy_scope_model_for_openai_only():
    for legacy_scope_model, should_change in (
        ("anthropic/claude-opus-4.6", True),
        ("openai/gpt-5.5", True),
        ("anthropic/claude-fable-5", True),
        ("anthropic::claude-fable-5", True),
        ("openai::gpt-5.6-terra", False),
    ):
        normalized, changed, changed_keys = apply_runtime_provider_defaults({
            "OPENAI_API_KEY": "sk-openai",
            "OUROBOROS_MODEL": "openai::gpt-5.6-terra",
            "OUROBOROS_MODEL_HEAVY": "openai::owner-selected-heavy",
            "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
            "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai::gpt-5.6-sol",
        "OUROBOROS_REVIEW_MODELS": "openai::gpt-5.6-terra,openai::gpt-5.5-mini",
        "OUROBOROS_SCOPE_REVIEW_MODEL": legacy_scope_model,
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai::gpt-5.6-terra",
    })

        assert changed is should_change
        assert changed_keys == (["OUROBOROS_SCOPE_REVIEW_MODEL"] if should_change else [])
        assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "openai::gpt-5.6-terra"


def test_apply_runtime_provider_defaults_migrates_prior_scope_default_on_general_path():
    """The shipped scope-review default moved openai/gpt-5.5 → anthropic/claude-fable-5
    (v6.55.0) → openai/gpt-5.6-terra (v6.82.0). An aggregator install whose SAVED scope
    value equals ANY old shipped default (never an explicit choice) must pick up the
    current default on upgrade; explicit lists and non-default values stay untouched."""
    for prior_default in ("openai/gpt-5.5", "anthropic/claude-fable-5", "anthropic::claude-fable-5"):
        normalized, changed, changed_keys = apply_runtime_provider_defaults({
            "OPENROUTER_API_KEY": "sk-or",
            "OUROBOROS_SCOPE_REVIEW_MODEL": prior_default,
            "OUROBOROS_SCOPE_REVIEW_MODELS": prior_default,
        })

        assert changed
        assert set(changed_keys) == {"OUROBOROS_SCOPE_REVIEW_MODEL", "OUROBOROS_SCOPE_REVIEW_MODELS"}
        assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "openai/gpt-5.6-terra"
        assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai/gpt-5.6-terra"

    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENROUTER_API_KEY": "sk-or",
        # Non-default single value and a deliberate multi-model list: preserved.
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5-pro",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash",
    })

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "openai/gpt-5.5-pro"
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai/gpt-5.5,google/gemini-3.5-flash"


def test_apply_runtime_provider_defaults_normalizes_anthropic_only_setup():
    """Legacy path: saved settings.json from older versions had claude-opus-4.6 —
    must still normalize to the Anthropic direct-provider prefix form.
    This guards backward compatibility for existing user installs."""
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "ANTHROPIC_API_KEY": "sk-ant",
        "OUROBOROS_MODEL": "anthropic/claude-opus-4.6",
        "OUROBOROS_MODEL_HEAVY": "anthropic/claude-opus-4.6",
        "OUROBOROS_MODEL_LIGHT": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
    })

    assert changed
    assert set(changed_keys) == {
        "OUROBOROS_MODEL",
        "OUROBOROS_MODEL_HEAVY",
        "OUROBOROS_MODEL_LIGHT",
        "OUROBOROS_MODEL_FALLBACKS",
        # v6.82.0: deep self-review is a per-provider slot too, so a direct-only
        # install never keeps an unreachable OpenRouter-form id for it.
        "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
    }
    assert normalized["OUROBOROS_MODEL"] == "anthropic::claude-opus-5"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "anthropic::claude-sonnet-5"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "anthropic::claude-sonnet-5"
    # ABI-10: retired comma keys are never introduced into settings.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in normalized

    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "ANTHROPIC_API_KEY": "sk-ant",
        "OUROBOROS_MODEL": "google/gemini-3.1-flash-lite",
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.1-flash-lite",
        "OUROBOROS_MODEL_LIGHT": "google/gemini-3.1-flash-lite",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_REVIEW_MODELS": (
            "openai/gpt-5.5,google/gemini-3.1-pro-preview,anthropic/claude-opus-4.6"
        ),
    })

    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"] == "anthropic::claude-opus-5"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "anthropic::claude-sonnet-5"
    assert normalized["OUROBOROS_REVIEW_MODELS"] == (
        "anthropic::claude-opus-5,"
        "anthropic::claude-opus-5,"
        "anthropic::claude-opus-5"
    )


def test_apply_runtime_provider_defaults_keeps_new_triad_on_openrouter():
    """v6.82.0 triad-expansion contract, OpenRouter side: with OpenRouter configured
    the shipped cross-provider triad and scope default pass through UNCHANGED (no
    direct-provider expansion, no prior-default remap of the CURRENT defaults)."""
    payload = dict(SETTINGS_DEFAULTS)
    payload["OPENROUTER_API_KEY"] = "sk-or"
    normalized, changed, changed_keys = apply_runtime_provider_defaults(payload)

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_MODEL"] == "google/gemini-3.8-flash"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai/gpt-5.6-luna"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai/gpt-5.6-luna"
    # ABI 7.0 (ABI-10): the comma keys are retired settings — a defaults-only
    # payload carries none and none may be introduced.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in normalized


def test_apply_runtime_provider_defaults_preserves_saved_outgoing_triad_on_openrouter():
    """A changed shipped default must not migrate an existing owner value."""
    outgoing = "openai/gpt-5.6-terra,google/gemini-3.6-flash,deepseek/deepseek-v4-pro"
    payload = dict(SETTINGS_DEFAULTS)
    payload["OPENROUTER_API_KEY"] = "sk-or"
    payload["OUROBOROS_REVIEW_MODELS"] = outgoing

    normalized, changed, changed_keys = apply_runtime_provider_defaults(payload)

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_REVIEW_MODELS"] == outgoing


def test_apply_runtime_provider_defaults_skips_non_official_or_custom_configs():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OPENAI_BASE_URL": "https://compat.example/v1",
        "OUROBOROS_MODEL": "custom-model",
    })

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_MODEL"] == "custom-model"


# --- Tests for Fix C (classify_runtime_provider_change) ---

from ouroboros.server_runtime import classify_runtime_provider_change


class TestClassifyRuntimeProviderChange:
    def test_direct_normalize_when_openrouter_absent(self):
        before = {"OPENAI_API_KEY": "sk-openai"}
        after = {"OPENAI_API_KEY": "sk-openai", "OUROBOROS_MODEL": "openai::gpt-5.5"}
        assert classify_runtime_provider_change(before, after) == "direct_normalize"

    def test_reverse_migrate_when_openrouter_added(self):
        before = {"OPENAI_API_KEY": "sk-openai"}
        after = {
            "OPENAI_API_KEY": "sk-openai",
            "OPENROUTER_API_KEY": "sk-or-v1-new",
            "OUROBOROS_MODEL": "openai::gpt-5.5",
        }
        assert classify_runtime_provider_change(before, after) == "reverse_migrate"

    def test_none_when_no_exclusive_provider_and_no_openrouter(self):
        before = {}
        after = {"OPENAI_COMPATIBLE_API_KEY": "compat-key"}
        assert classify_runtime_provider_change(before, after) == "none"

    def test_direct_normalize_for_anthropic_only(self):
        before = {"ANTHROPIC_API_KEY": "sk-ant"}
        after = {"ANTHROPIC_API_KEY": "sk-ant", "OUROBOROS_MODEL": "anthropic::claude-opus-4-8"}
        assert classify_runtime_provider_change(before, after) == "direct_normalize"

    def test_reverse_migrate_for_anthropic_plus_openrouter(self):
        before = {"ANTHROPIC_API_KEY": "sk-ant"}
        after = {
            "ANTHROPIC_API_KEY": "sk-ant",
            "OPENROUTER_API_KEY": "sk-or-v1-new",
            "OUROBOROS_MODEL": "anthropic::claude-opus-4-8",
        }
        assert classify_runtime_provider_change(before, after) == "reverse_migrate"

    def test_direct_normalize_for_openai_only_no_change_marker(self):
        # classify only looks at 'after' state — before is unused but accepted
        before = {}
        after = {"OPENAI_API_KEY": "sk-openai"}
        assert classify_runtime_provider_change(before, after) == "direct_normalize"

    def test_none_when_both_openai_and_anthropic(self):
        # Two direct providers → not exclusive → none
        before = {}
        after = {"OPENAI_API_KEY": "sk-openai", "ANTHROPIC_API_KEY": "sk-ant"}
        assert classify_runtime_provider_change(before, after) == "none"


class TestSettingsSaveWarningContract:
    """Verify the warning-gate contract used by server.py::api_settings_post.

    server.py does:
        current, provider_defaults_changed, _ = apply_runtime_provider_defaults(current)
        if provider_defaults_changed:
            change_kind = classify_runtime_provider_change(old_settings, current)
            if change_kind == "direct_normalize":
                warnings.append("Normalized direct-provider routing ...")

    We test this logic directly — (1) direct normalization should produce a warning,
    (2) adding OpenRouter back should NOT produce a warning.
    """

    def _simulate_save_warning(self, old_settings: dict, new_settings: dict) -> list[str]:
        """Simulate the api_settings_post warning logic."""
        from ouroboros.server_runtime import apply_runtime_provider_defaults
        current, provider_defaults_changed, _ = apply_runtime_provider_defaults(dict(new_settings))
        warnings: list[str] = []
        if provider_defaults_changed:
            change_kind = classify_runtime_provider_change(old_settings, current)
            if change_kind == "direct_normalize":
                warnings.append(
                    "Normalized direct-provider routing because OpenRouter is not configured."
                )
        return warnings

    def test_direct_normalization_produces_warning(self):
        # First save with only OpenAI — direct normalization fires, warning expected
        old = {}
        new = {"OPENAI_API_KEY": "sk-openai"}
        warnings = self._simulate_save_warning(old, new)
        assert len(warnings) == 1
        assert "Normalized" in warnings[0]

    def test_adding_openrouter_back_produces_no_warning(self):
        # User was in OpenAI-only mode, then adds OpenRouter —
        # apply_runtime_provider_defaults returns no changes (OpenRouter present),
        # so provider_defaults_changed is False and the warning block is never reached.
        old = {"OPENAI_API_KEY": "sk-openai", "OUROBOROS_MODEL": "openai::gpt-5.5"}
        new = {"OPENAI_API_KEY": "sk-openai", "OPENROUTER_API_KEY": "sk-or-v1", "OUROBOROS_MODEL": "openai::gpt-5.5"}
        warnings = self._simulate_save_warning(old, new)
        assert warnings == []


def test_apply_runtime_provider_defaults_cloudru_only_elevates_to_direct():
    """A Cloud.ru-only user (no OpenRouter/OpenAI/Anthropic) must get cloudru::
    direct routing for main/code AND for the review/scope reviewer slots, so they
    can fully use Ouroboros (incl. passing tri-model review) with only a Cloud.ru key."""
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr-key",
    })
    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"].startswith("cloudru::")
    assert "OUROBOROS_MODEL_HEAVY" not in normalized
    # ABI-10: reviewer reachability resolves at READ time, not by introducing
    # retired settings keys.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized


def test_cloudru_only_review_models_resolve_at_read_time(monkeypatch):
    triad, scope = _read_time_review_models(monkeypatch, {
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr-key",
        "OUROBOROS_MODEL": "cloudru::zai-org/GLM-4.6",
    })
    assert triad and all(m.startswith("cloudru::") for m in triad)
    assert scope and all(m.startswith("cloudru::") for m in scope)


def test_apply_runtime_provider_defaults_minimax_only_uses_current_models():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "MINIMAX_API_KEY": "minimax-key",
    })

    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"] == "minimax::MiniMax-M3"
    assert "OUROBOROS_MODEL_HEAVY" not in normalized
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "minimax::MiniMax-M2.7"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "minimax::MiniMax-M2.7"
    # Deep self-review stays empty: MiniMax guarantees only a 512K window floor,
    # below the 1M target deep review sizes against (clear-instead-of-fill).
    assert not normalized.get("OUROBOROS_MODEL_DEEP_SELF_REVIEW")
    # ABI-10: retired comma keys are never introduced into settings.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized


def test_minimax_only_review_models_resolve_at_read_time(monkeypatch):
    triad, scope = _read_time_review_models(monkeypatch, {
        "MINIMAX_API_KEY": "minimax-key", "OUROBOROS_MODEL": "minimax::MiniMax-M3",
    })
    assert triad == ["minimax::MiniMax-M3", "minimax::MiniMax-M2.7", "minimax::MiniMax-M2.7"]
    assert scope and all(m.startswith("minimax::") for m in scope)


def test_apply_runtime_provider_defaults_cloudru_migrates_populated_shipped_defaults():
    """The realistic save path: a Cloud.ru-only user whose settings already carry
    shipped non-cloudru defaults, including the immediately outgoing v6.104 Main
    and Light values. Every active model/review slot migrates to cloudru::;
    the product-authored Heavy default is discarded rather than becoming an
    owner-custom actor."""
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    normalized, changed, _ = apply_runtime_provider_defaults({
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr-key",
        "OUROBOROS_MODEL": "x-ai/grok-4.5",
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_LIGHT": "google/gemini-3.6-flash",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.8",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.5",
    })
    assert changed
    assert normalized["OUROBOROS_MODEL"].startswith("cloudru::")
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert all(m.startswith("cloudru::") for m in normalized["OUROBOROS_REVIEW_MODELS"].split(","))
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"].startswith("cloudru::")
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"].startswith("cloudru::")


def test_apply_runtime_provider_defaults_gigachat_only_elevates_to_direct():
    """A GigaChat-only user (user/password auth, no other provider) must get
    gigachat:: direct routing for main/code AND for the review/scope reviewer
    slots — exercises the exclusive-direct path that previously KeyError'd because
    gigachat lacked a _DIRECT_PROVIDER_AUTO_DEFAULTS entry."""
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "GIGACHAT_USER": "user",
        "GIGACHAT_PASSWORD": "pass",
    })
    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"].startswith("gigachat::")
    assert normalized["OUROBOROS_MODEL"] == "gigachat::GigaChat-2-Max"
    assert "OUROBOROS_MODEL_HEAVY" not in normalized
    # ABI-10: retired comma keys are never introduced into settings.
    assert "OUROBOROS_REVIEW_MODELS" not in normalized
    assert "OUROBOROS_SCOPE_REVIEW_MODEL" not in normalized


def test_gigachat_only_review_models_resolve_at_read_time(monkeypatch):
    triad, scope = _read_time_review_models(monkeypatch, {
        "GIGACHAT_USER": "user", "GIGACHAT_PASSWORD": "pass",
        "OUROBOROS_MODEL": "gigachat::GigaChat-2-Max",
    })
    assert triad and all(m.startswith("gigachat::") for m in triad)
    assert scope and all(m.startswith("gigachat::") for m in scope)


def test_apply_runtime_provider_defaults_gigachat_credentials_migrates_shipped_defaults():
    """A GigaChat-only user (authorization key) whose settings still carry the
    shipped (non-gigachat) defaults: active Main/Light and every review/scope
    reviewer slot migrate; the shipped Heavy is not misclassified as custom."""
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    normalized, changed, _ = apply_runtime_provider_defaults({
        "GIGACHAT_CREDENTIALS": "base64-key",
        "OUROBOROS_MODEL": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_LIGHT": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.8",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.5",
    })
    assert changed
    assert normalized["OUROBOROS_MODEL"].startswith("gigachat::")
    assert normalized["OUROBOROS_MODEL"] == "gigachat::GigaChat-2-Max"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == ""
    assert all(m.startswith("gigachat::") for m in normalized["OUROBOROS_REVIEW_MODELS"].split(","))
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"].startswith("gigachat::")
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"].startswith("gigachat::")


def test_gigachat_defaults_are_reachable_for_personal_and_legal_scopes():
    """Ultra is Freemium-personal only; the shipped Max default works for every
    supported auth scope instead of stranding B2B/CORP fresh installs."""
    for scope in ("GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"):
        normalized, changed, _keys = apply_runtime_provider_defaults({
            "GIGACHAT_CREDENTIALS": "base64-key",
            "GIGACHAT_SCOPE": scope,
        })
        assert changed
        assert normalized["OUROBOROS_MODEL"] == "gigachat::GigaChat-2-Max"
        assert normalized["GIGACHAT_SCOPE"] == scope


def test_local_only_install_keeps_light_and_fallback_on_the_local_main_route():
    """Provider Independence: with NO remote credential, the shipped remote Light /
    Fallback defaults are cleared so both slots inherit Main (the local route). An
    explicit choice is never touched, and a remote install keeps the defaults."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    local_only = {
        "OUROBOROS_MODEL": SETTINGS_DEFAULTS["OUROBOROS_MODEL"],
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_LIGHT": SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"],
        "OUROBOROS_MODEL_FALLBACKS": SETTINGS_DEFAULTS["OUROBOROS_MODEL_FALLBACKS"],
        "LOCAL_MODEL_SOURCE": "repo/model.gguf",
        "USE_LOCAL_MAIN": True,
    }
    out, changed, keys = apply_runtime_provider_defaults(dict(local_only))
    assert out["OUROBOROS_MODEL_HEAVY"] == ""
    assert out["OUROBOROS_MODEL_LIGHT"] == ""
    assert out["OUROBOROS_MODEL_FALLBACKS"] == ""
    assert changed and "OUROBOROS_MODEL_LIGHT" in keys

    explicit = dict(local_only, OUROBOROS_MODEL_LIGHT="local-light")
    assert apply_runtime_provider_defaults(explicit)[0]["OUROBOROS_MODEL_LIGHT"] == "local-light"

    remote = dict(local_only, OPENROUTER_API_KEY="sk-or-test")
    kept = apply_runtime_provider_defaults(remote)[0]["OUROBOROS_MODEL_LIGHT"]
    assert kept == SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]

    local_heavy = dict(
        local_only,
        USE_LOCAL_HEAVY=True,
        OUROBOROS_MODEL_HEAVY="google/gemini-3.5-flash",
    )
    assert apply_runtime_provider_defaults(local_heavy)[0]["OUROBOROS_MODEL_HEAVY"] == (
        "google/gemini-3.5-flash"
    )

    # UPGRADE path: a local-only settings file still holding the immediately
    # outgoing shipped values is just as unreachable and must inherit local Main.
    upgraded = {
        "OUROBOROS_MODEL": "local-main",
        "OUROBOROS_MODEL_LIGHT": "google/gemini-3.6-flash",
        "OUROBOROS_MODEL_FALLBACKS": "openai/gpt-5.6-luna",
        "LOCAL_MODEL_SOURCE": "repo/model.gguf",
        "USE_LOCAL_MAIN": True,
    }
    out_upgraded = apply_runtime_provider_defaults(dict(upgraded))[0]
    assert out_upgraded["OUROBOROS_MODEL_LIGHT"] == ""
    assert out_upgraded["OUROBOROS_MODEL_FALLBACKS"] == ""


def test_local_only_install_keeps_a_slot_the_owner_routed_to_local(monkeypatch):
    """The shipped-default clearing exists to drop UNREACHABLE remote defaults. A slot
    explicitly routed to local is reachable — and for Fallbacks the model string is the
    only chain entry, so clearing it would delete the lane rather than inherit Main."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.server_runtime import _clear_shipped_defaults_for_local_only

    settings = {
        "LOCAL_MODEL_SOURCE": "lmstudio",
        "USE_LOCAL_MAIN": False,
        "USE_LOCAL_FALLBACK": True,
        "OUROBOROS_MODEL_LIGHT": SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"],
        "OUROBOROS_MODEL_FALLBACKS": SETTINGS_DEFAULTS["OUROBOROS_MODEL_FALLBACKS"],
    }
    changed = _clear_shipped_defaults_for_local_only(settings)

    assert "OUROBOROS_MODEL_FALLBACKS" not in changed
    assert settings["OUROBOROS_MODEL_FALLBACKS"] == SETTINGS_DEFAULTS["OUROBOROS_MODEL_FALLBACKS"], (
        "a locally-routed fallback lane must keep a concrete chain entry"
    )
    # ...while a slot with NO local flag is still cleared as unreachable.
    assert "OUROBOROS_MODEL_LIGHT" in changed
    assert settings["OUROBOROS_MODEL_LIGHT"] == ""


def test_direct_only_install_gets_a_reachable_deep_review_model():
    """The deep self-review slot ships an OpenRouter-form default. A direct-only
    install has no OpenRouter credential, so that id is unreachable — the
    provider-defaults path must migrate this slot like the other four."""
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.provider_models import ANTHROPIC_DIRECT_DEFAULTS, OPENAI_DIRECT_DEFAULTS

    for key, expected in (
        ("OPENAI_API_KEY", OPENAI_DIRECT_DEFAULTS["deep_self_review"]),
        ("ANTHROPIC_API_KEY", ANTHROPIC_DIRECT_DEFAULTS["deep_self_review"]),
    ):
        # Driven from the FULL shipped defaults dict, which is what a real install
        # carries: the OpenRouter-form deep default is migrated into a direct
        # spelling before the comparison, so the guard must recognise BOTH forms or
        # the slot silently keeps a `-pro` id that 404s on api.openai.com.
        populated = dict(SETTINGS_DEFAULTS)
        populated[key] = "sk-test"
        normalized, changed, changed_keys = apply_runtime_provider_defaults(populated)
        assert changed
        assert "OUROBOROS_MODEL_DEEP_SELF_REVIEW" in changed_keys
        assert normalized["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] == expected
        assert "/" not in expected.split("::", 1)[1]

    # A sub-floor provider gets NO auto-filled deep slot: deep review sizes against
    # a fixed 1M window; Cloud.ru/GigaChat are documented below that floor, and
    # MiniMax guarantees only a 512K minimum ("up to 1M"), so filling the slot
    # would advertise a review doomed to overflow its route.
    # A sub-floor provider must end up with NO deep slot at all: the shipped
    # OpenRouter-form default is unreachable for it, and auto-filling its own model
    # would advertise a review doomed to overflow the 1M window deep review sizes
    # against. Driven from a FULL defaults dict and the credentials each provider
    # is actually recognised by (GigaChat authenticates with GIGACHAT_CREDENTIALS,
    # not an API-key spelling — a wrong key silently skips the direct path
    # entirely and makes this assertion vacuous).
    for creds in ({"CLOUDRU_FOUNDATION_MODELS_API_KEY": "sk-test"},
                  {"GIGACHAT_CREDENTIALS": "Z2ln-test"},
                  {"MINIMAX_API_KEY": "minimax-test"}):
        populated = dict(SETTINGS_DEFAULTS)
        populated.update(creds)
        normalized, _changed, changed_keys = apply_runtime_provider_defaults(populated)
        assert "OUROBOROS_MODEL" in changed_keys, "the ordinary slots still normalize"
        assert not normalized.get("OUROBOROS_MODEL_DEEP_SELF_REVIEW"), normalized.get(
            "OUROBOROS_MODEL_DEEP_SELF_REVIEW"
        )
        # An EXPLICIT owner choice in that slot is never cleared.
        explicit = dict(populated)
        explicit["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] = "openrouter::owner/pick"
        kept, _c, _k = apply_runtime_provider_defaults(explicit)
        assert kept["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] == "openrouter::owner/pick"


def test_upgraded_direct_install_migrates_the_prior_deep_review_default():
    """An upgraded install still carries v6.81's OpenRouter-form deep value, which
    is just as unreachable without an OpenRouter credential as the other slots."""
    from ouroboros.provider_models import ANTHROPIC_DIRECT_DEFAULTS, OPENAI_DIRECT_DEFAULTS

    for key, expected in (
        ("OPENAI_API_KEY", OPENAI_DIRECT_DEFAULTS["deep_self_review"]),
        ("ANTHROPIC_API_KEY", ANTHROPIC_DIRECT_DEFAULTS["deep_self_review"]),
    ):
        normalized, _changed, changed_keys = apply_runtime_provider_defaults({
            key: "sk-test", "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/gpt-5.5-pro",
        })
        assert "OUROBOROS_MODEL_DEEP_SELF_REVIEW" in changed_keys
        assert normalized["OUROBOROS_MODEL_DEEP_SELF_REVIEW"] == expected


def test_a_fresh_local_first_install_still_authors_safety_light(tmp_path, monkeypatch):
    """LOCAL-FIRST: LOCAL_MODEL_SOURCE is present at first launch, so the
    pre-wizard normalization DOES flag changes (it clears the unreachable remote
    Light/Fallback defaults). The launcher must not persist that on a fresh
    install — the save would create the settings file and silently cost the
    wizard its safety-light authorship. Pins the launcher's actual decision:
    persist only when the file already exists."""
    import ouroboros.config as cfg
    from ouroboros.settings_setup_contract import wizard_authors_safety_light

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=False)

    fresh = dict(SETTINGS_DEFAULTS)
    fresh["LOCAL_MODEL_SOURCE"] = "lmstudio"
    _normalized, changed, changed_keys = apply_runtime_provider_defaults(fresh)
    assert changed, "local-first normalization genuinely flags changes"
    assert "OUROBOROS_MODEL_LIGHT" in changed_keys

    # The launcher's persistence decision, verbatim: changed AND file exists.
    assert not (changed and settings_path.exists()), (
        "a fresh install must not persist the pre-wizard normalization"
    )
    assert wizard_authors_safety_light() is True

    settings_path.write_text("{}", encoding="utf-8")
    assert (changed and settings_path.exists()), "an existing install still persists it"
    assert wizard_authors_safety_light() is False

    # ...and the pre-onboarding normalizers implement at least that decision. BOTH
    # have since gone further and persist nothing at all, which satisfies the
    # fresh-install rule by construction rather than by a carve-out that has to be
    # got right (behavioural coverage: tests/test_onboarding_host.py).
    repo = cfg.pathlib.Path(__file__).parent.parent
    launcher_host = (repo / "ouroboros" / "launcher_onboarding.py").read_text(encoding="utf-8")
    server_src = (repo / "server.py").read_text(encoding="utf-8")
    assert "save_settings(" not in launcher_host
    assert "save_settings(" not in server_src
