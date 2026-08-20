"""Runtime mode plumbing: the config helpers and the onboarding validation.

This module owns the settings defaults and the frozen mode tuple, the
``get_runtime_mode`` clamp/case/default behaviour, the skills-repo path helper, the
env propagation of both keys, the legacy invalid mode a load clamps, and the
onboarding payload validation that accepts each mode and rejects an unknown one.

The publishing surfaces, the registry gating, the run_shell gating, the light-mode
skill-payload short form and the repair-mode confinement were split verbatim into
``tests/test_runtime_mode_surfaces.py``,
``tests/test_runtime_mode_registry_gating.py``,
``tests/test_runtime_mode_shell_gating.py``,
``tests/test_runtime_mode_skill_payload.py`` and
``tests/test_runtime_mode_repair_confinement.py``; the registry, git-repo and
skill-payload builders they share live in ``tests/_runtime_mode_core_shared.py``.

Security-critical self-elevation lives in the ``test_runtime_mode_elevation.py``
family: the ``save_settings`` chokepoint there, the ``_data_write`` fence in
``tests/test_runtime_mode_data_write.py``, the owner endpoints in
``tests/test_runtime_mode_owner_endpoints.py``, mode authorship in
``tests/test_runtime_mode_authorship.py`` and the deterministic command guards in
``tests/test_runtime_mode_write_guards.py``.
"""
from __future__ import annotations

import os

import pytest



# ===========================================================================
# Part 1: config.py defaults + helpers + env propagation
# ===========================================================================


def test_settings_defaults_include_phase2_keys():
    from ouroboros.config import SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert SETTINGS_DEFAULTS["OUROBOROS_SKILLS_REPO_PATH"] == ""
    assert SETTINGS_DEFAULTS["OUROBOROS_MODEL"] == "google/gemini-3.7-flash"
    # Empty role slots inherit Main; Light and Fallback use Luna explicitly.
    assert SETTINGS_DEFAULTS["OUROBOROS_MODEL_HEAVY"] == ""
    assert SETTINGS_DEFAULTS["OUROBOROS_MODEL_VISION"] == ""
    assert SETTINGS_DEFAULTS["OUROBOROS_MODEL_CONSCIOUSNESS"] == ""
    assert SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"] == "openai/gpt-5.6-luna"
    assert SETTINGS_DEFAULTS["OUROBOROS_MODEL_FALLBACKS"] == "openai/gpt-5.6-luna"
    assert (
        SETTINGS_DEFAULTS["OUROBOROS_MODEL_DEEP_SELF_REVIEW"]
        == "openai/gpt-5.6-sol-pro"
    )
    assert SETTINGS_DEFAULTS["CLAUDE_CODE_MODEL"] == "claude-sonnet-5"
    assert SETTINGS_DEFAULTS["TOTAL_BUDGET"] == 200.0
    assert SETTINGS_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"] == 50.0


def test_llm_internal_fallbacks_follow_shipped_model_defaults(monkeypatch):
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.llm import DEFAULT_LIGHT_MODEL, LLMClient

    monkeypatch.delenv("OUROBOROS_MODEL", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_HEAVY", raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)
    client = LLMClient.__new__(LLMClient)

    assert DEFAULT_LIGHT_MODEL == SETTINGS_DEFAULTS["OUROBOROS_MODEL_LIGHT"]
    assert client.default_model() == SETTINGS_DEFAULTS["OUROBOROS_MODEL"]
    assert client.available_models() == [SETTINGS_DEFAULTS["OUROBOROS_MODEL"]]


def test_valid_runtime_modes_is_frozen_tuple():
    from ouroboros.config import VALID_RUNTIME_MODES

    assert VALID_RUNTIME_MODES == ("light", "advanced", "pro")


@pytest.mark.parametrize("mode", ["light", "advanced", "pro"])
def test_get_runtime_mode_accepts_all_three(mode, monkeypatch):
    from ouroboros.config import get_runtime_mode

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    assert get_runtime_mode() == mode


def test_get_runtime_mode_clamps_unknown_value(monkeypatch):
    from ouroboros.config import get_runtime_mode

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "ULTRA")
    assert get_runtime_mode() == "advanced"


def test_get_runtime_mode_is_case_insensitive(monkeypatch):
    from ouroboros.config import get_runtime_mode

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "Pro")
    assert get_runtime_mode() == "pro"


def test_get_runtime_mode_defaults_when_unset(monkeypatch):
    from ouroboros.config import get_runtime_mode

    monkeypatch.delenv("OUROBOROS_RUNTIME_MODE", raising=False)
    assert get_runtime_mode() == "advanced"


def test_get_skills_repo_path_defaults_to_empty(monkeypatch):
    from ouroboros.config import get_skills_repo_path

    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    assert get_skills_repo_path() == ""


def test_get_skills_repo_path_expands_home(monkeypatch):
    from ouroboros.config import get_skills_repo_path

    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "~/Ouroboros/skills")
    expanded = get_skills_repo_path()
    assert expanded.startswith(os.path.expanduser("~"))
    assert expanded.endswith(os.path.join("Ouroboros", "skills"))


def test_apply_settings_to_env_propagates_phase2_keys(monkeypatch):
    from ouroboros.config import SETTINGS_DEFAULTS, apply_settings_to_env

    monkeypatch.delenv("OUROBOROS_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)

    settings = dict(SETTINGS_DEFAULTS)
    settings["OUROBOROS_RUNTIME_MODE"] = "light"
    settings["OUROBOROS_SKILLS_REPO_PATH"] = "/tmp/skills"

    apply_settings_to_env(settings)

    assert os.environ["OUROBOROS_RUNTIME_MODE"] == "light"
    assert os.environ["OUROBOROS_SKILLS_REPO_PATH"] == "/tmp/skills"


def test_normalize_runtime_mode_clamps_unknown_inputs():
    from ouroboros.config import normalize_runtime_mode

    assert normalize_runtime_mode("light") == "light"
    assert normalize_runtime_mode("ADVANCED") == "advanced"
    assert normalize_runtime_mode("Pro") == "pro"
    assert normalize_runtime_mode("turbo") == "advanced"
    assert normalize_runtime_mode("") == "advanced"
    assert normalize_runtime_mode(None) == "advanced"
    assert normalize_runtime_mode(123) == "advanced"


def test_load_settings_clamps_legacy_invalid_runtime_mode(tmp_path, monkeypatch):
    """Read-path normalization: a pre-existing settings.json containing
    an invalid runtime mode must be clamped at load time so /api/settings
    (GET) and the onboarding bootstrap cannot echo stale invalid values.
    """
    import importlib
    import json

    import ouroboros.config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({
            "OUROBOROS_RUNTIME_MODE": "turbo",
            "OUROBOROS_SKILLS_REPO_PATH": "   ",
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(settings_path))
    monkeypatch.delenv("OUROBOROS_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)

    try:
        cfg_reloaded = importlib.reload(cfg)
        loaded = cfg_reloaded.load_settings()
        assert loaded["OUROBOROS_RUNTIME_MODE"] == "advanced"
        assert loaded["OUROBOROS_SKILLS_REPO_PATH"] == ""
    finally:
        os.environ.pop("OUROBOROS_SETTINGS_PATH", None)
        importlib.reload(cfg)


# ===========================================================================
# Part 2: onboarding_wizard validation
# ===========================================================================


def _onboarding_payload_with_runtime(mode: str | None = None, skills_path: str = ""):
    from ouroboros.config import SETTINGS_DEFAULTS

    payload = {
        "OPENROUTER_API_KEY": "sk-or-v1-" + "a" * 30,
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "TOTAL_BUDGET": 10,
        "OUROBOROS_PER_TASK_COST_USD": 20,
        "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
        "LOCAL_MODEL_SOURCE": "",
        "LOCAL_MODEL_FILENAME": "",
        "LOCAL_MODEL_CONTEXT_LENGTH": SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"],
        "LOCAL_MODEL_N_GPU_LAYERS": -1,
        "LOCAL_MODEL_CHAT_FORMAT": "",
        "LOCAL_ROUTING_MODE": "cloud",
        "OUROBOROS_MODEL": "anthropic/claude-opus-4.6",
        "OUROBOROS_MODEL_HEAVY": "anthropic/claude-opus-4.6",
        "OUROBOROS_MODEL_LIGHT": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_MODEL_FALLBACKS": "anthropic/claude-sonnet-4.6",
        "OUROBOROS_SKILLS_REPO_PATH": skills_path,
    }
    if mode is not None:
        payload["OUROBOROS_RUNTIME_MODE"] = mode
    return payload


def test_prepare_onboarding_settings_defaults_runtime_mode_when_missing():
    from ouroboros.onboarding_wizard import prepare_onboarding_settings

    payload = _onboarding_payload_with_runtime(mode=None)
    prepared, error = prepare_onboarding_settings(payload, {})
    assert error is None, error
    assert prepared["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert prepared["OUROBOROS_SKILLS_REPO_PATH"] == ""


@pytest.mark.parametrize("mode", ["light", "advanced", "pro"])
def test_prepare_onboarding_settings_accepts_each_runtime_mode(mode):
    from ouroboros.onboarding_wizard import prepare_onboarding_settings

    payload = _onboarding_payload_with_runtime(mode=mode)
    prepared, error = prepare_onboarding_settings(payload, {})
    assert error is None, error
    assert prepared["OUROBOROS_RUNTIME_MODE"] == mode


def test_prepare_onboarding_settings_rejects_unknown_runtime_mode():
    from ouroboros.onboarding_wizard import prepare_onboarding_settings

    payload = _onboarding_payload_with_runtime(mode="turbo")
    prepared, error = prepare_onboarding_settings(payload, {})
    assert prepared == {}
    assert error is not None
    assert "runtime mode" in error.lower()


def test_prepare_onboarding_settings_persists_skills_repo_path():
    from ouroboros.onboarding_wizard import prepare_onboarding_settings

    payload = _onboarding_payload_with_runtime(mode="advanced", skills_path="~/skills-dev")
    prepared, error = prepare_onboarding_settings(payload, {})
    assert error is None, error
    assert prepared["OUROBOROS_SKILLS_REPO_PATH"] == "~/skills-dev"


def test_onboarding_bootstrap_exposes_runtime_mode():
    from ouroboros.onboarding_wizard import build_onboarding_html

    html = build_onboarding_html(
        {"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_SKILLS_REPO_PATH": "/opt/skills"}
    )
    assert '"runtimeMode": "pro"' in html
    assert '"skillsRepoPath": "/opt/skills"' in html
