from ouroboros.server_runtime import (
    apply_runtime_provider_defaults,
    has_startup_ready_provider,
    needs_local_model_autostart,
)
from ouroboros.config import SETTINGS_DEFAULTS


def test_has_startup_ready_provider_accepts_any_remote_key_or_local_routing():
    assert has_startup_ready_provider({"OPENROUTER_API_KEY": "sk-or-test"})
    assert has_startup_ready_provider({"OPENAI_API_KEY": "sk-openai"})
    assert has_startup_ready_provider({"ANTHROPIC_API_KEY": "sk-ant"})
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
        "OUROBOROS_REVIEW_MODELS",
        "OUROBOROS_SCOPE_REVIEW_MODEL",
        "OUROBOROS_SCOPE_REVIEW_MODELS",
    }
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.4-mini"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai::gpt-5.4-mini"

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
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.4-mini"
    assert normalized["OUROBOROS_REVIEW_MODELS"] == (
        "openai::gpt-5.5,openai::gpt-5.4-mini,openai::gpt-5.4-mini"
    )
    # v4.39.0: direct-provider fallback now seeds `[main, light, light]` —
    # 3 commit-triad slots (preserving the documented 3-reviewer contract)
    # with 2 unique models (so `plan_task`'s quorum gate passes). Replaces
    # the old `[main] * 3` fallback that broke `plan_task` first-run.
    assert normalized["OUROBOROS_REVIEW_MODELS"] == (
        "openai::gpt-5.5,openai::gpt-5.4-mini,openai::gpt-5.4-mini"
    )
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai::gpt-5.5"

    payload = dict(SETTINGS_DEFAULTS)
    payload["OPENAI_API_KEY"] = "sk-openai"
    normalized, changed, changed_keys = apply_runtime_provider_defaults(payload)

    assert changed
    assert "OUROBOROS_MODEL" in changed_keys
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.4-mini"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai::gpt-5.4-mini"


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
        "OUROBOROS_REVIEW_MODELS",
        "OUROBOROS_SCOPE_REVIEW_MODEL",
        "OUROBOROS_SCOPE_REVIEW_MODELS",
    }
    assert normalized["OUROBOROS_MODEL"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == "openai::gpt-5.5"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "openai::gpt-5.4-mini"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "openai::gpt-5.4-mini"
    # v6.36.0 (D4): an explicit provider-matching review list is honored EXACTLY
    # (1 model = 1 slot — a loud single_reviewer_no_diversity degraded mode), not
    # silently expanded to [main, light, light]. Expansion fires only when the
    # configured list is empty or contains foreign (non-provider) models.
    assert normalized["OUROBOROS_REVIEW_MODELS"] == "openai::gpt-5.5"


def test_apply_runtime_provider_defaults_keeps_explicit_official_openai_review_models():
    # All model slots already correct; scope review model is unset → gets migrated to main.
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "openai::gpt-5.5",
        "OUROBOROS_MODEL_HEAVY": "openai::gpt-5.5",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
        "OUROBOROS_REVIEW_MODELS": "openai::gpt-5.5,openai::gpt-5.5-mini",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai::gpt-5.5",  # already in direct format
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai::gpt-5.5",  # already in direct format
    })

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_REVIEW_MODELS"] == "openai::gpt-5.5,openai::gpt-5.5-mini"


def test_apply_runtime_provider_defaults_preserves_duplicate_scope_slots_for_openai():
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENAI_API_KEY": "sk-openai",
        "OUROBOROS_MODEL": "openai::gpt-5.5",
        "OUROBOROS_MODEL_HEAVY": "openai::gpt-5.5",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
        "OUROBOROS_REVIEW_MODELS": "openai::gpt-5.5,openai::gpt-5.5,openai::gpt-5.5",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai::gpt-5.5",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai::gpt-5.5,openai::gpt-5.5,openai::gpt-5.5",
    })

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "openai::gpt-5.5,openai::gpt-5.5,openai::gpt-5.5"


def test_apply_runtime_provider_defaults_preserves_current_opus47_defaults_with_openrouter():
    current_openrouter = "anthropic/claude-opus-" + "4.7"
    current_claude_code = "claude-opus-" + "4-7[1m]"
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENROUTER_API_KEY": "sk-or",
        "OUROBOROS_MODEL": current_openrouter,
        "OUROBOROS_MODEL_HEAVY": current_openrouter,
        "OUROBOROS_REVIEW_MODELS": f"openai/gpt-5.5,{current_openrouter}",
        "CLAUDE_CODE_MODEL": current_claude_code,
    })

    assert not changed
    assert changed_keys == []
    assert normalized["OUROBOROS_MODEL"] == current_openrouter
    assert normalized["OUROBOROS_MODEL_HEAVY"] == current_openrouter
    assert normalized["OUROBOROS_REVIEW_MODELS"] == f"openai/gpt-5.5,{current_openrouter}"
    assert normalized["CLAUDE_CODE_MODEL"] == current_claude_code


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
        ("openai::gpt-5.5", False),
    ):
        normalized, changed, changed_keys = apply_runtime_provider_defaults({
            "OPENAI_API_KEY": "sk-openai",
            "OUROBOROS_MODEL": "openai::gpt-5.5",
            "OUROBOROS_MODEL_HEAVY": "openai::gpt-5.5",
            "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
            "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
        "OUROBOROS_REVIEW_MODELS": "openai::gpt-5.5,openai::gpt-5.5-mini",
        "OUROBOROS_SCOPE_REVIEW_MODEL": legacy_scope_model,
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai::gpt-5.5",
    })

        assert changed is should_change
        assert changed_keys == (["OUROBOROS_SCOPE_REVIEW_MODEL"] if should_change else [])
        assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "openai::gpt-5.5"


def test_apply_runtime_provider_defaults_migrates_prior_scope_default_on_general_path():
    """v6.55.0: the shipped scope-review default moved openai/gpt-5.5 →
    anthropic/claude-fable-5. An aggregator install whose SAVED scope value equals
    the old default (never an explicit choice) must pick up the new default on
    upgrade (scope fable-5 cumulative-review finding); explicit lists and
    non-default values stay untouched."""
    normalized, changed, changed_keys = apply_runtime_provider_defaults({
        "OPENROUTER_API_KEY": "sk-or",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.5",
    })

    assert changed
    assert set(changed_keys) == {"OUROBOROS_SCOPE_REVIEW_MODEL", "OUROBOROS_SCOPE_REVIEW_MODELS"}
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "anthropic/claude-fable-5"
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "anthropic/claude-fable-5"

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
        "OUROBOROS_REVIEW_MODELS",
        "OUROBOROS_SCOPE_REVIEW_MODEL",
        "OUROBOROS_SCOPE_REVIEW_MODELS",
    }
    assert normalized["OUROBOROS_MODEL"] == "anthropic::claude-opus-4-8"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == "anthropic::claude-opus-4-8"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "anthropic::claude-sonnet-4-6"
    assert normalized["OUROBOROS_MODEL_FALLBACKS"] == "anthropic::claude-sonnet-4-6"
    # v4.39.0: `[main, light, light]` — 3 commit-triad slots, 2 unique.
    assert normalized["OUROBOROS_REVIEW_MODELS"] == (
        "anthropic::claude-opus-4-8,"
        "anthropic::claude-sonnet-4-6,"
        "anthropic::claude-sonnet-4-6"
    )
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"] == "anthropic::claude-opus-4-8"
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"] == "anthropic::claude-opus-4-8"

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
    assert normalized["OUROBOROS_MODEL"] == "anthropic::claude-opus-4-8"
    assert normalized["OUROBOROS_MODEL_HEAVY"] == "anthropic::claude-opus-4-8"
    assert normalized["OUROBOROS_MODEL_LIGHT"] == "anthropic::claude-sonnet-4-6"
    assert normalized["OUROBOROS_REVIEW_MODELS"] == (
        "anthropic::claude-opus-4-8,"
        "anthropic::claude-sonnet-4-6,"
        "anthropic::claude-sonnet-4-6"
    )


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
    assert normalized["OUROBOROS_MODEL_HEAVY"].startswith("cloudru::")
    assert all(m.startswith("cloudru::") for m in normalized["OUROBOROS_REVIEW_MODELS"].split(","))
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"].startswith("cloudru::")


def test_apply_runtime_provider_defaults_cloudru_migrates_populated_shipped_defaults():
    """The realistic save path: a Cloud.ru-only user whose settings already carry
    the shipped (non-cloudru) defaults. main/code AND every review/scope reviewer
    slot must still migrate to cloudru:: so no slot points at a provider with no key."""
    from ouroboros.server_runtime import apply_runtime_provider_defaults

    normalized, changed, _ = apply_runtime_provider_defaults({
        "CLOUDRU_FOUNDATION_MODELS_API_KEY": "cr-key",
        "OUROBOROS_MODEL": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_HEAVY": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_LIGHT": "google/gemini-3.5-flash",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.8",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "openai/gpt-5.5",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "openai/gpt-5.5",
    })
    assert changed
    assert normalized["OUROBOROS_MODEL"].startswith("cloudru::")
    assert normalized["OUROBOROS_MODEL_HEAVY"].startswith("cloudru::")
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
    assert normalized["OUROBOROS_MODEL_HEAVY"].startswith("gigachat::")
    assert all(m.startswith("gigachat::") for m in normalized["OUROBOROS_REVIEW_MODELS"].split(","))
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"].startswith("gigachat::")


def test_apply_runtime_provider_defaults_gigachat_credentials_migrates_shipped_defaults():
    """A GigaChat-only user (authorization key) whose settings still carry the
    shipped (non-gigachat) defaults: main/code AND every review/scope reviewer slot
    must migrate to gigachat:: so no slot points at a provider with no key."""
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
    assert normalized["OUROBOROS_MODEL_HEAVY"].startswith("gigachat::")
    assert all(m.startswith("gigachat::") for m in normalized["OUROBOROS_REVIEW_MODELS"].split(","))
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODEL"].startswith("gigachat::")
    assert normalized["OUROBOROS_SCOPE_REVIEW_MODELS"].startswith("gigachat::")
