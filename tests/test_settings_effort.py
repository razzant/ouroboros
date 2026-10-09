"""Tests for effort, review models, and review enforcement settings."""
import json
import os
from ouroboros.config import (
    SETTINGS_DEFAULTS,
    apply_settings_to_env,
    resolve_effort,
    get_review_enforcement,
    get_task_review_mode,
    get_context_mode,
    get_image_input_mode,
    get_vision_caption_timeout_sec,
    get_vision_model,
    review_model_uses_local,
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
    assert SETTINGS_DEFAULTS.get("OUROBOROS_EFFORT_CONSCIOUSNESS") == ""  # empty = the Task / Chat effort
    # The review surface efforts are retired settings (review pool: effort is a field of
    # the reviewer row); the resolver keeps its own "high" default for callers.
    for retired in ("OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW", "OUROBOROS_EFFORT_DEEP_SELF_REVIEW"):
        assert retired not in SETTINGS_DEFAULTS


def test_review_effort_default_carriers_stay_in_sync():
    """The owner-facing fallback must not drift from config/API defaults.

    A reviewer is a catalog row marked Reviewer, and the catalog editor is the
    owner-facing carrier: a marked row with no effort of its own (and no compound
    session effort) states that it reviews at the pool default, the former
    OUROBOROS_EFFORT_REVIEW default."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    editor = (root / "web" / "modules" / "subagents_settings.js").read_text(encoding="utf-8")
    assert "export const REVIEW_POOL_DEFAULT_EFFORT = 'high';" in editor
    assert "reviews at ${REVIEW_POOL_DEFAULT_EFFORT} effort" in editor
    # The surface effort keys are retired (review pool: effort lives on the reviewer
    # row); the read seam migrates them, so they are no shipped default any more.
    assert "OUROBOROS_EFFORT_REVIEW" not in SETTINGS_DEFAULTS
    assert "OUROBOROS_EFFORT_SCOPE_REVIEW" not in SETTINGS_DEFAULTS
    from ouroboros.config import REVIEW_POOL_DEFAULT_EFFORT
    from ouroboros.reviewer_slot_config import ConfiguredReviewerSlot, row_effort

    bare = ConfiguredReviewerSlot(slot_id="r", kind="api", target_id="openai/gpt-5.6-terra")
    assert row_effort(bare) == REVIEW_POOL_DEFAULT_EFFORT == "high"


_RETIRED_EFFORT_KEYS = ("OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW", "OUROBOROS_EFFORT_DEEP_SELF_REVIEW")


def test_an_exported_retired_review_effort_key_is_inert_for_the_deep_review(monkeypatch):
    """The deep review's Main row carries no effort of its own, so it reviews at the
    pool default; the lane-era surface keys are retired, and one exported in the
    process environment (a benchmark container, an operator shell) changes nothing —
    ``resolve_effort`` no longer has a branch that reads them."""
    import inspect

    from ouroboros import settings_scales
    from ouroboros.config import REVIEW_POOL_DEFAULT_EFFORT
    from ouroboros.deep_self_review import main_review_row
    from ouroboros.reviewer_slot_config import row_effort

    for key in _RETIRED_EFFORT_KEYS:
        monkeypatch.setenv(key, "low")
    assert row_effort(main_review_row()) == REVIEW_POOL_DEFAULT_EFFORT
    source = inspect.getsource(settings_scales.resolve_effort)
    assert not any(key in source for key in _RETIRED_EFFORT_KEYS)
    assert "deep_self_review" not in source and "scope_review" not in source


def test_review_models_default_in_config():
    """ABI 7.0 (ABI-10): the comma key is RETIRED from the settings vocabulary;
    the shipped triad default lives in OPENROUTER_REVIEW_DEFAULTS."""
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS, RETIRED_SETTING_KEYS

    assert "OUROBOROS_REVIEW_MODELS" not in SETTINGS_DEFAULTS
    assert "OUROBOROS_REVIEW_MODELS" in RETIRED_SETTING_KEYS
    assert list(OPENROUTER_REVIEW_DEFAULTS["triad"]) == [
        "google/gemini-3.8-flash",
        "openai/gpt-5.6-terra",
        "anthropic/claude-opus-5",
    ]


def test_review_enforcement_default_in_config():
    """OUROBOROS_REVIEW_ENFORCEMENT defaults to advisory."""
    assert SETTINGS_DEFAULTS.get("OUROBOROS_REVIEW_ENFORCEMENT") == "advisory"


def test_scope_review_and_task_review_defaults_in_config():
    from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS, RETIRED_SETTING_KEYS

    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in SETTINGS_DEFAULTS
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" in RETIRED_SETTING_KEYS
    assert OPENROUTER_REVIEW_DEFAULTS["scope"] == ("openai/gpt-5.6-terra",)
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
    assert "modelRolesHost('settings-model-roles')" in settings_ui
    assert slots["vision"]["settingsInputId"] == "s-model-vision"


def test_auto_grant_reviewed_skills_default_in_config():
    assert SETTINGS_DEFAULTS.get("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS") == "true"


# ---------------------------------------------------------------------------
# Factory review pools
# ---------------------------------------------------------------------------


def _factory_pool_models(doc: dict) -> list:
    """The review POOL a settings document gets at the factory (PR-3): the
    exclusive-provider panel is minted into the catalog once, never multiplied
    at read time by the pool reader."""
    from ouroboros.subscription_install_presets import factory_review_rows

    return [row["route"]["target_id"] for row in factory_review_rows(doc)]


def test_factory_pool_repeats_main_in_openai_only_mode():
    """The OpenAI-only profile mints three independent Main catalog rows."""
    assert _factory_pool_models({"OPENAI_API_KEY": "configured", "OUROBOROS_MODEL": "openai::gpt-5.6-terra"}) == [
        "openai::gpt-5.6-terra",
        "openai::gpt-5.6-terra",
        "openai::gpt-5.6-terra",
    ]


def test_factory_pool_repeats_main_in_anthropic_only_mode():
    """The Anthropic-only profile repeats even an explicit provider Main."""
    assert _factory_pool_models({"ANTHROPIC_API_KEY": "sk-ant", "OUROBOROS_MODEL": "anthropic::claude-opus-4-6"}) == [
        "anthropic::claude-opus-4-6",
        "anthropic::claude-opus-4-6",
        "anthropic::claude-opus-4-6",
    ]


def test_factory_pool_routes_to_gigachat_in_gigachat_only_mode(monkeypatch):
    """v6.14.0: GigaChat joins the direct-provider review panel. A GigaChat-only
    install (no other provider) gets a gigachat:: review pool, never an empty one
    or an unconfigured foreign provider — the single-isolated-provider invariant
    (docs/DEVELOPMENT.md "Provider Independence"). GIGACHAT_DIRECT_DEFAULTS uses the
    universally available GigaChat-2-Max for every slot, so the quorum-safe panel is
    [main, main, main] — three catalog rows (PR-3)."""
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "giga-creds")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    monkeypatch.setenv("OUROBOROS_MODEL", "gigachat::GigaChat-2-Max")

    assert _factory_pool_models({"GIGACHAT_CREDENTIALS": "giga-creds", "OUROBOROS_MODEL": "gigachat::GigaChat-2-Max"}) == [
        "gigachat::GigaChat-2-Max",
        "gigachat::GigaChat-2-Max",
        "gigachat::GigaChat-2-Max",
    ]


def test_from_zero_local_only_review_slots_inherit_main_and_stay_local(monkeypatch):
    """A fresh local-only install must not contact shipped remote reviewers."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL", "OPENAI_COMPATIBLE_API_KEY",
        "OPENAI_COMPATIBLE_BASE_URL", "CLOUDRU_FOUNDATION_MODELS_API_KEY",
        "GIGACHAT_CREDENTIALS", "GIGACHAT_USER", "GIGACHAT_PASSWORD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_MODEL_SOURCE", "owner/local.gguf")
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")
    monkeypatch.setenv("OUROBOROS_MODEL", "owner/local-main")
    monkeypatch.setenv(
        "OUROBOROS_REVIEW_MODELS",
        "openai/gpt-5.6-luna,google/gemini-3.6-flash,anthropic/claude-sonnet-5",
    )

    # The factory pool is the shipped panel's three seats on the local Main (three
    # independent runs, quorum 2 of 3); every review slot built from it runs on the
    # local lane.
    assert _factory_pool_models({"USE_LOCAL_MAIN": "true", "LOCAL_MODEL_SOURCE": "owner/local.gguf",
                                 "OUROBOROS_MODEL": "owner/local-main"}) == ["owner/local-main"] * 3
    assert review_model_uses_local("owner/local-main") is True

    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.subscription_install_presets import factory_review_rows

    rows = factory_review_rows({"USE_LOCAL_MAIN": "true", "LOCAL_MODEL_SOURCE": "owner/local.gguf",
                                "OUROBOROS_MODEL": "owner/local-main"})
    slots = review_pool_slots({"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": rows})})
    assert [slot.model for slot in slots] == ["owner/local-main"] * 3  # three runs of Main, quorum 2 of 3
    assert all(slot.use_local for slot in slots)


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


def test_apply_settings_to_env_includes_context_mode(monkeypatch, tmp_path):
    """apply_settings_to_env propagates an AUTHORED OUROBOROS_CONTEXT_MODE to the env.

    Authored means STORED IN settings.json, and it overrides a contradicting env value. The
    default that ``load_settings`` substitutes for an absent key is not an owner decision, so
    projecting it would clobber a value a benchmark launcher forwarded on purpose — see
    ``test_env_forwarded_modes_survive_the_documented_startup_path``.
    """
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_CONTEXT_MODE": "low"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    # apply_settings_to_env writes os.environ directly for ~122 keys: rely on the autouse conftest environ snapshot.
    os.environ["OUROBOROS_CONTEXT_MODE"] = "max"

    apply_settings_to_env({"OUROBOROS_CONTEXT_MODE": "low"})
    assert os.environ.get("OUROBOROS_CONTEXT_MODE") == "low"


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


def test_apply_settings_ignores_the_retired_review_models_key(monkeypatch):
    """ABI-10: the retired comma key is IGNORED by apply_settings_to_env — a ghost
    value neither reaches the env plane nor is replaced by a projected floor
    (PR-3 removed the lane projection); the catalog alone selects the pool."""
    monkeypatch.delenv("OUROBOROS_REVIEW_MODELS", raising=False)
    monkeypatch.delenv("OUROBOROS_SUBAGENTS", raising=False)
    from ouroboros.reviewer_slot_config import review_pool_slots
    from tests.review_pool_rosters import packet_pool

    settings = {"OUROBOROS_REVIEW_MODELS": "ghost/value",
                "OUROBOROS_SUBAGENTS": packet_pool(["vendor/selected"])}
    apply_settings_to_env(settings)
    assert "OUROBOROS_REVIEW_MODELS" not in os.environ
    assert [slot.model for slot in review_pool_slots()] == ["vendor/selected"]


def test_apply_settings_clears_review_enforcement_restores_default(monkeypatch):
    """Clearing OUROBOROS_REVIEW_ENFORCEMENT restores the default in env."""
    settings = {"OUROBOROS_REVIEW_ENFORCEMENT": ""}
    apply_settings_to_env(settings)
    env_val = os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT", "")
    assert env_val == SETTINGS_DEFAULTS["OUROBOROS_REVIEW_ENFORCEMENT"]
    assert get_review_enforcement() == "advisory"


def test_apply_settings_clears_task_review_restores_default_and_ignores_retired_scope_key(monkeypatch):
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODELS", raising=False)
    monkeypatch.delenv("OUROBOROS_SCOPE_REVIEW_MODEL", raising=False)
    settings = {"OUROBOROS_SCOPE_REVIEW_MODELS": "ghost/value", "OUROBOROS_TASK_REVIEW_MODE": ""}
    apply_settings_to_env(settings)
    assert "OUROBOROS_SCOPE_REVIEW_MODELS" not in os.environ
    assert os.environ.get("OUROBOROS_TASK_REVIEW_MODE") == SETTINGS_DEFAULTS["OUROBOROS_TASK_REVIEW_MODE"]


# ---------------------------------------------------------------------------
# apply_settings_to_env propagation
# ---------------------------------------------------------------------------

def test_apply_settings_to_env_includes_effort_keys(monkeypatch, tmp_path):
    """apply_settings_to_env propagates all effort keys."""
    settings = {
        "OUROBOROS_EFFORT_TASK": "low",
        "OUROBOROS_EFFORT_EVOLUTION": "medium",
        # Retired review-lane efforts in a stale settings dict are ghosts too (the read
        # seam migrates them into the reviewer rows): apply must NOT export them.
        "OUROBOROS_EFFORT_REVIEW": "high",
        "OUROBOROS_EFFORT_SCOPE_REVIEW": "low",
        "OUROBOROS_EFFORT_CONSCIOUSNESS": "none",
        # ABI-10: retired comma keys in a stale settings dict are ghosts —
        # apply must NOT export them (asserted below).
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
    assert os.environ.get("OUROBOROS_EFFORT_REVIEW") is None
    assert os.environ.get("OUROBOROS_EFFORT_SCOPE_REVIEW") is None
    assert os.environ.get("OUROBOROS_EFFORT_CONSCIOUSNESS") == "none"
    # ABI-10: the retired comma-list INPUT is ignored — the env carries neither the
    # retired value nor a projected floor (the lane projection left with the lanes).
    assert os.environ.get("OUROBOROS_REVIEW_MODELS") is None
    assert os.environ.get("OUROBOROS_REVIEW_ENFORCEMENT") == "advisory"
    assert os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS") is None
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
