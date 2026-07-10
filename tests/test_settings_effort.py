"""Tests for effort, review models, and review enforcement settings."""
import json
import os
from ouroboros.config import (
    SETTINGS_DEFAULTS,
    apply_settings_to_env,
    resolve_effort,
    get_review_models,
    get_review_enforcement,
    get_scope_review_models,
    get_task_review_mode,
    get_context_mode,
    get_image_input_mode,
    get_vision_caption_timeout_sec,
    get_vision_model,
)


# ---------------------------------------------------------------------------
# Legacy env var backward compat
# ---------------------------------------------------------------------------

def test_initial_effort_default(monkeypatch):
    """Default effort is 'medium' when env var not set."""
    monkeypatch.delenv("OUROBOROS_EFFORT_TASK", raising=False)
    assert resolve_effort("task") == "medium"


def test_initial_effort_valid_values(monkeypatch):
    """Valid effort values pass through unchanged via OUROBOROS_EFFORT_TASK."""
    for effort in ("none", "low", "medium", "high"):
        monkeypatch.setenv("OUROBOROS_EFFORT_TASK", effort)
        assert resolve_effort("task") == effort


def test_initial_effort_invalid_falls_back_to_medium(monkeypatch):
    """Invalid effort values fall back to 'medium'."""
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "extreme")
    assert resolve_effort("task") == "medium"


# ---------------------------------------------------------------------------
# New per-type defaults in SETTINGS_DEFAULTS
# ---------------------------------------------------------------------------

def test_effort_defaults_in_config():
    """All effort keys have correct defaults in SETTINGS_DEFAULTS."""
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_TASK") == "medium"
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_EVOLUTION") == "high"
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_REVIEW") == "medium"
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_SCOPE_REVIEW") == "high"
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_DEEP_SELF_REVIEW") == "high"
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_CONSCIOUSNESS") == "high"


def test_deep_self_review_effort_slot(monkeypatch):
    monkeypatch.delenv("OUROBOROS_EFFORT_DEEP_SELF_REVIEW", raising=False)
    assert resolve_effort("deep_self_review") == "high"
    monkeypatch.setenv("OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "medium")
    assert resolve_effort("deep_self_review") == "medium"
    monkeypatch.setenv("OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "extreme")
    assert resolve_effort("deep_self_review") == "high"


def test_review_models_default_in_config():
    """OUROBOROS_REVIEW_MODELS has a default value in config."""
    val = SETTINGS_DEFAULTS.get("OUROBOROS_REVIEW_MODELS", "")
    assert val  # non-empty
    models = [m.strip() for m in val.split(",") if m.strip()]
    assert len(models) >= 2  # quorum requires at least 2


def test_review_enforcement_default_in_config():
    """OUROBOROS_REVIEW_ENFORCEMENT defaults to advisory."""
    assert SETTINGS_DEFAULTS.get("OUROBOROS_REVIEW_ENFORCEMENT") == "advisory"


def test_scope_review_and_task_review_defaults_in_config():
    assert SETTINGS_DEFAULTS.get("OUROBOROS_SCOPE_REVIEW_MODELS") == "anthropic/claude-fable-5"
    assert SETTINGS_DEFAULTS.get("OUROBOROS_TASK_REVIEW_MODE") == "auto"


def test_vision_settings_defaults_and_setup_contract(monkeypatch):
    from ouroboros.settings_setup_contract import build_setup_contract

    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.5")
    monkeypatch.delenv("OUROBOROS_MODEL_VISION", raising=False)
    monkeypatch.delenv("OUROBOROS_IMAGE_INPUT_MODE", raising=False)
    assert get_vision_model() == "openai/gpt-5.5"
    assert get_image_input_mode() == "auto"
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-2.5-pro")
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    assert get_vision_model() == "google/gemini-2.5-pro"
    assert get_image_input_mode() == "caption"
    monkeypatch.setenv("OUROBOROS_VISION_CAPTION_TIMEOUT_SEC", "17")
    assert get_vision_caption_timeout_sec() == 17
    payload = build_setup_contract()
    steps = {step["id"]: step for step in payload["steps"]}
    assert steps["models"]["railCopy"] == "model slots"
    slots = {slot["slot"]: slot for slot in payload["modelSlots"]}
    assert slots["vision"]["settingKey"] == "OUROBOROS_MODEL_VISION"
    assert slots["vision"]["settingsToggleId"] == ""
    import pathlib
    settings_ui = (pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    assert "'s-model-vision', ''," in settings_ui


def test_auto_grant_reviewed_skills_default_in_config():
    assert SETTINGS_DEFAULTS.get("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS") == "true"


# ---------------------------------------------------------------------------
# get_review_models() — single source of truth
# ---------------------------------------------------------------------------

def test_get_review_models_default(monkeypatch):
    """get_review_models() returns the config default when env is unset."""
    monkeypatch.delenv("OUROBOROS_REVIEW_MODELS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    models = get_review_models()
    assert isinstance(models, list)
    assert len(models) >= 2
    assert all("/" in m for m in models)  # valid OpenRouter model IDs


def test_get_review_models_custom(monkeypatch):
    """get_review_models() returns custom models when env is set."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "a/b,c/d")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    models = get_review_models()
    assert models == ["a/b", "c/d"]


def test_get_review_models_empty_env_falls_back_to_default(monkeypatch):
    """get_review_models() falls back to default when env is empty string."""
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    models = get_review_models()
    # Must return the default, not an empty list
    assert len(models) >= 2
    assert models == [m.strip() for m in SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"].split(",") if m.strip()]


def test_get_review_models_falls_back_to_main_light_light_in_openai_only_mode(monkeypatch):
    """v4.39.0: direct-provider fallback returns [main, light, light] (3 slots,
    2 unique) instead of the legacy [main]*N so both commit triad and
    plan_task have a quorum-safe reviewer list out of the box. The light slot
    picks up the provider default (OPENAI_DIRECT_DEFAULTS['light'] =
    openai::gpt-5.4-mini) when OUROBOROS_MODEL_LIGHT is not explicitly set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.5")
    monkeypatch.setenv(
        "OUROBOROS_REVIEW_MODELS",
        "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.6",
    )

    models = get_review_models()

    assert models == [
        "openai::gpt-5.5",
        "openai::gpt-5.4-mini",
        "openai::gpt-5.4-mini",
    ]


def test_get_review_models_does_not_apply_openai_only_fallback_with_compatible_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compat.example/v1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    monkeypatch.delenv("GIGACHAT_USER", raising=False)
    monkeypatch.delenv("GIGACHAT_PASSWORD", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.5")
    monkeypatch.setenv(
        "OUROBOROS_REVIEW_MODELS",
        "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.6",
    )

    models = get_review_models()

    assert models == ["openai/gpt-5.5", "google/gemini-3.5-flash", "anthropic/claude-opus-4.6"]


def test_get_review_models_preserves_explicit_official_openai_list(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "openai::gpt-5.5")
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "openai/gpt-5.5,openai/gpt-4.1")

    models = get_review_models()

    assert models == ["openai::gpt-5.5", "openai::gpt-4.1"]


def test_get_review_models_falls_back_to_main_light_light_in_anthropic_only_mode(monkeypatch):
    """v4.39.0: same direct-provider fallback as OpenAI — [main, light, light]
    with light = ANTHROPIC_DIRECT_DEFAULTS['light'] = anthropic::claude-sonnet-4-6
    when OUROBOROS_MODEL_LIGHT is not explicitly set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "anthropic::claude-opus-4-6")
    monkeypatch.setenv(
        "OUROBOROS_REVIEW_MODELS",
        "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.6",
    )

    models = get_review_models()

    assert models == [
        "anthropic::claude-opus-4-6",
        "anthropic::claude-sonnet-4-6",
        "anthropic::claude-sonnet-4-6",
    ]


def test_get_review_models_and_scope_route_to_gigachat_in_gigachat_only_mode(monkeypatch):
    """v6.14.0: GigaChat joins the direct-provider review fallback. A GigaChat-only
    env (no other provider) must route the commit triad AND the scope reviewer to
    gigachat:: models, never to an empty list or an unconfigured foreign provider —
    the single-isolated-provider invariant (docs/DEVELOPMENT.md "Provider
    Independence"). GIGACHAT_DIRECT_DEFAULTS uses GigaChat-3-Ultra for every slot,
    so the quorum-safe fallback degrades to [main, main, main]."""
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "giga-creds")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "gigachat::GigaChat-3-Ultra")
    monkeypatch.setenv(
        "OUROBOROS_REVIEW_MODELS",
        "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.8",
    )
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "openai/gpt-5.5")

    review_models = get_review_models()
    scope_models = get_scope_review_models()

    assert review_models == [
        "gigachat::GigaChat-3-Ultra",
        "gigachat::GigaChat-3-Ultra",
        "gigachat::GigaChat-3-Ultra",
    ]
    assert scope_models and all(m.startswith("gigachat::") for m in scope_models)


def test_get_review_enforcement_default(monkeypatch):
    """get_review_enforcement() returns the config default when env is unset."""
    monkeypatch.delenv("OUROBOROS_REVIEW_ENFORCEMENT", raising=False)
    assert get_review_enforcement() == "advisory"


def test_get_review_enforcement_custom(monkeypatch):
    """get_review_enforcement() accepts advisory and blocking."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    assert get_review_enforcement() == "advisory"
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    assert get_review_enforcement() == "blocking"


def test_get_review_enforcement_invalid_falls_back(monkeypatch):
    """Unknown values fall back to advisory (the default)."""
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "strictest")
    assert get_review_enforcement() == "advisory"


def test_get_scope_review_models_preserves_duplicate_slots(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "model/a, model/a, model/b")
    assert get_scope_review_models() == ["model/a", "model/a", "model/b"]


def test_get_scope_review_models_falls_back_to_singular(monkeypatch):
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODELS", "")
    monkeypatch.setenv("OUROBOROS_SCOPE_REVIEW_MODEL", "legacy/scope")
    assert get_scope_review_models() == ["legacy/scope"]


def test_get_task_review_mode_clamps_invalid(monkeypatch):
    monkeypatch.delenv("OUROBOROS_TASK_REVIEW_MODE", raising=False)
    assert get_task_review_mode() == "auto"
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "required")
    assert get_task_review_mode() == "required"
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "blocking")
    assert get_task_review_mode() == "auto"


def test_context_mode_default_in_config():
    """OUROBOROS_CONTEXT_MODE defaults to max (today's behavior)."""
    assert SETTINGS_DEFAULTS.get("OUROBOROS_CONTEXT_MODE") == "max"


def test_get_context_mode_clamps_invalid(monkeypatch):
    """get_context_mode() clamps to the closed low/max enum (default max)."""
    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE", raising=False)
    assert get_context_mode() == "max"
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    assert get_context_mode() == "low"
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "MAX")
    assert get_context_mode() == "max"
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "ultra")
    assert get_context_mode() == "max"


def test_apply_settings_to_env_includes_context_mode(monkeypatch):
    """apply_settings_to_env propagates OUROBOROS_CONTEXT_MODE to the env."""
    apply_settings_to_env({"OUROBOROS_CONTEXT_MODE": "low"})
    assert os.environ.get("OUROBOROS_CONTEXT_MODE") == "low"
    os.environ.pop("OUROBOROS_CONTEXT_MODE", None)


def test_get_auto_grant_enabled(monkeypatch, tmp_path):
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "missing-settings.json", raising=True)
    monkeypatch.delenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", raising=False)
    # New SSOT default is "true": absent settings file + absent env → enabled.
    assert cfg.get_auto_grant_enabled() is True
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true")
    assert cfg.get_auto_grant_enabled() is True
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")
    assert cfg.get_auto_grant_enabled() is False


def test_get_auto_grant_enabled_prefers_settings_file(monkeypatch, tmp_path):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    monkeypatch.setenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "false")

    assert cfg.get_auto_grant_enabled() is True


def test_apply_settings_clears_review_models_restores_default(monkeypatch):
    """Clearing OUROBOROS_REVIEW_MODELS in settings restores the default in env."""
    # Simulate user clearing the field in Settings UI (empty string)
    settings = {"OUROBOROS_REVIEW_MODELS": ""}
    apply_settings_to_env(settings)
    # env var should be the default, not empty
    env_val = os.environ.get("OUROBOROS_REVIEW_MODELS", "")
    assert env_val == SETTINGS_DEFAULTS["OUROBOROS_REVIEW_MODELS"]
    # get_review_models() should also return correct defaults
    assert len(get_review_models()) >= 2


def test_apply_settings_clears_review_enforcement_restores_default(monkeypatch):
    """Clearing OUROBOROS_REVIEW_ENFORCEMENT restores the default in env."""
    settings = {"OUROBOROS_REVIEW_ENFORCEMENT": ""}
    apply_settings_to_env(settings)
    env_val = os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT", "")
    assert env_val == SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"]
    assert get_review_enforcement() == "advisory"


def test_apply_settings_clears_task_and_scope_review_restores_default(monkeypatch):
    settings = {"OUROBOROS_SCOPE_REVIEW_MODELS": "", "OUROBOROS_SCOPE_REVIEW_MODEL": "", "OUROBOROS_TASK_REVIEW_MODE": ""}
    apply_settings_to_env(settings)
    assert os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS") == SETTINGS_DEFAULTS["OUROBOROS_SCOPE_REVIEW_MODELS"]
    assert os.environ.get("OUROBOROS_TASK_REVIEW_MODE") == SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"]


# ---------------------------------------------------------------------------
# apply_settings_to_env propagation
# ---------------------------------------------------------------------------

def test_apply_settings_to_env_includes_effort_keys(monkeypatch, tmp_path):
    """apply_settings_to_env propagates all effort keys."""
    settings = {
        "OUROBOROS_EFFORT_TASK": "low",
        "OUROBOROS_EFFORT_EVOLUTION": "medium",
        "OUROBOROS_EFFORT_REVIEW": "high",
        "OUROBOROS_EFFORT_SCOPE_REVIEW": "low",
        "OUROBOROS_EFFORT_CONSCIOUSNESS": "none",
        "OUROBOROS_REVIEW_MODELS": "model-a,model-b",
        "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
        "OUROBOROS_SCOPE_REVIEW_MODELS": "scope-a,scope-b",
        "OUROBOROS_TASK_REVIEW_MODE": "required",
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true",
        "OUROBOROS_RETURN_REASONING": "",
    }
    apply_settings_to_env(settings)
    assert os.environ.get("OUROBOROS_EFFORT_TASK") == "low"
    assert os.environ.get("OUROBOROS_EFFORT_EVOLUTION") == "medium"
    assert os.environ.get("OUROBOROS_EFFORT_REVIEW") == "high"
    assert os.environ.get("OUROBOROS_EFFORT_SCOPE_REVIEW") == "low"
    assert os.environ.get("OUROBOROS_EFFORT_CONSCIOUSNESS") == "none"
    assert os.environ.get("OUROBOROS_REVIEW_MODELS") == "model-a,model-b"
    assert os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT") == "advisory"
    assert os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS") == "scope-a,scope-b"
    assert os.environ.get("OUROBOROS_TASK_REVIEW_MODE") == "required"
    assert os.environ.get("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS") == "true"
    assert os.environ.get("OUROBOROS_RETURN_REASONING") == ""
    # cleanup
    for k in ("OUROBOROS_EFFORT_TASK", "OUROBOROS_EFFORT_EVOLUTION",
              "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
              "OUROBOROS_EFFORT_CONSCIOUSNESS",
              "OUROBOROS_REVIEW_MODELS", "OUROBOROS_REVIEW_ENFORCEMENT",
              "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_TASK_REVIEW_MODE",
              "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "OUROBOROS_RETURN_REASONING"):
        os.environ.pop(k, None)

    import ouroboros.config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_RETURN_REASONING": True}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    monkeypatch.setenv("OUROBOROS_RETURN_REASONING", "")

    loaded = cfg.load_settings()
    assert loaded["OUROBOROS_RETURN_REASONING"] == ""
    cfg.apply_settings_to_env(loaded)
    assert os.environ.get("OUROBOROS_RETURN_REASONING") == ""
