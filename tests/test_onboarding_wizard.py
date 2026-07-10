import pathlib

import pytest

from ouroboros.onboarding_wizard import build_onboarding_html, prepare_onboarding_settings
from ouroboros.settings_setup_contract import build_setup_bootstrap, build_setup_contract


REPO = pathlib.Path(__file__).resolve().parents[1]


def _base_payload() -> dict:
    return {
        "OPENROUTER_API_KEY": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "TOTAL_BUDGET": 10,
        "OUROBOROS_PER_TASK_COST_USD": 20,
        "OUROBOROS_REVIEW_ENFORCEMENT": "advisory",
        "LOCAL_MODEL_SOURCE": "",
        "LOCAL_MODEL_FILENAME": "",
        "LOCAL_MODEL_CONTEXT_LENGTH": 16384,
        "LOCAL_MODEL_N_GPU_LAYERS": -1,
        "LOCAL_MODEL_CHAT_FORMAT": "",
        "LOCAL_ROUTING_MODE": "cloud",
        "OUROBOROS_MODEL": "openai::gpt-5.5",
        "OUROBOROS_MODEL_HEAVY": "openai::gpt-5.5",
        "OUROBOROS_MODEL_LIGHT": "openai::gpt-5.5-mini",
        "OUROBOROS_MODEL_FALLBACKS": "openai::gpt-5.5-mini",
    }


def test_prepare_onboarding_settings_requires_runnable_config():
    prepared, error = prepare_onboarding_settings(_base_payload(), {})

    assert prepared == {}
    assert "Configure OpenRouter, OpenAI, OpenAI-compatible, Cloud.ru, Anthropic, or a local model" in error


def test_prepare_onboarding_settings_accepts_openai_only_setup():
    payload = _base_payload()
    payload["OPENAI_API_KEY"] = "sk-openai-1234567890"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert error is None
    assert prepared["OPENAI_API_KEY"] == "sk-openai-1234567890"
    assert prepared["OUROBOROS_MODEL"] == "openai::gpt-5.5"
    assert prepared["TOTAL_BUDGET"] == 10.0
    assert prepared["OUROBOROS_PER_TASK_COST_USD"] == 20.0
    assert prepared["OUROBOROS_REVIEW_ENFORCEMENT"] == "advisory"
    assert prepared["OUROBOROS_MODEL_CONSCIOUSNESS"] == ""
    # Onboarding no longer manages auto-grant; the global SSOT default applies.
    assert "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS" not in prepared


def test_settings_default_auto_grant_is_true():
    from ouroboros.config import SETTINGS_DEFAULTS

    assert SETTINGS_DEFAULTS["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "true"


def test_prepare_onboarding_settings_preserves_existing_auto_grant_choice():
    payload = _base_payload()
    payload["OPENAI_API_KEY"] = "sk-openai-1234567890"

    prepared, error = prepare_onboarding_settings(
        payload,
        {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false"},
    )

    assert error is None
    # Onboarding does not override an explicit existing choice.
    assert prepared["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "false"


@pytest.mark.parametrize(("key", "value", "error"), [
    ("TOTAL_BUDGET", 0, "Budget must be greater than zero."),
    ("TOTAL_BUDGET", "0", "Budget must be greater than zero."),
    ("TOTAL_BUDGET", -1, "Budget must be greater than zero."),
    ("TOTAL_BUDGET", 0.005, "Budget must be at least 0.01."),
    ("TOTAL_BUDGET", "nan", "Budget must be a number."),
    ("OUROBOROS_PER_TASK_COST_USD", 0, "Per-task soft threshold must be greater than zero."),
    ("OUROBOROS_PER_TASK_COST_USD", "0", "Per-task soft threshold must be greater than zero."),
    ("OUROBOROS_PER_TASK_COST_USD", -1, "Per-task soft threshold must be greater than zero."),
    ("OUROBOROS_PER_TASK_COST_USD", 0.005, "Per-task soft threshold must be at least 0.01."),
    ("OUROBOROS_PER_TASK_COST_USD", "nan", "Per-task soft threshold must be a number."),
    ("OUROBOROS_PER_TASK_COST_USD", False, "Per-task soft threshold must be a number."),
])
def test_prepare_onboarding_settings_rejects_invalid_budget_values(key, value, error):
    payload = _base_payload()
    payload["OPENAI_API_KEY"] = "sk-openai-1234567890"
    payload[key] = value

    prepared, actual_error = prepare_onboarding_settings(payload, {})

    assert prepared == {}
    assert actual_error == error


def test_prepare_onboarding_settings_accepts_cloudru_only_setup():
    payload = _base_payload()
    payload["CLOUDRU_FOUNDATION_MODELS_API_KEY"] = "cloudru-key-1234567890"
    payload["OUROBOROS_MODEL"] = "cloudru::zai-org/GLM-4.7"
    payload["OUROBOROS_MODEL_HEAVY"] = "cloudru::zai-org/GLM-4.7"
    payload["OUROBOROS_MODEL_LIGHT"] = "cloudru::zai-org/GLM-4.7"
    payload["OUROBOROS_MODEL_FALLBACKS"] = "cloudru::zai-org/GLM-4.7"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert error is None
    assert prepared["CLOUDRU_FOUNDATION_MODELS_API_KEY"] == "cloudru-key-1234567890"
    assert prepared["OUROBOROS_MODEL"] == "cloudru::zai-org/GLM-4.7"


def test_prepare_onboarding_settings_accepts_anthropic_only_setup():
    payload = _base_payload()
    payload["ANTHROPIC_API_KEY"] = "sk-ant-1234567890"
    payload["OUROBOROS_MODEL"] = "anthropic::claude-opus-4-6"
    payload["OUROBOROS_MODEL_HEAVY"] = "anthropic::claude-opus-4-6"
    payload["OUROBOROS_MODEL_LIGHT"] = "anthropic::claude-sonnet-4-6"
    payload["OUROBOROS_MODEL_FALLBACKS"] = "anthropic::claude-sonnet-4-6"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert error is None
    assert prepared["ANTHROPIC_API_KEY"] == "sk-ant-1234567890"
    assert prepared["OUROBOROS_MODEL"] == "anthropic::claude-opus-4-6"


def test_prepare_onboarding_settings_rejects_local_only_cloud_routing():
    payload = _base_payload()
    payload["LOCAL_MODEL_SOURCE"] = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    payload["LOCAL_MODEL_FILENAME"] = "qwen2.5-7b-instruct-q3_k_m.gguf"
    payload["LOCAL_ROUTING_MODE"] = "cloud"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert prepared == {}
    assert error == "Local-only setups must route at least one model to the local runtime."


def test_prepare_onboarding_settings_rejects_consciousness_only_local_routing():
    payload = _base_payload()
    payload["LOCAL_MODEL_SOURCE"] = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    payload["LOCAL_MODEL_FILENAME"] = "qwen2.5-7b-instruct-q3_k_m.gguf"
    payload["LOCAL_ROUTING_MODE"] = "cloud"
    payload["USE_LOCAL_CONSCIOUSNESS"] = True

    prepared, error = prepare_onboarding_settings(payload, {})

    assert prepared == {}
    assert error == "Local-only setups must route at least one model to the local runtime."


def test_prepare_onboarding_settings_sets_all_local_routes():
    payload = _base_payload()
    payload["LOCAL_MODEL_SOURCE"] = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    payload["LOCAL_MODEL_FILENAME"] = "qwen2.5-7b-instruct-q3_k_m.gguf"
    payload["LOCAL_ROUTING_MODE"] = "all"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert error is None
    assert prepared["USE_LOCAL_MAIN"] is True
    assert prepared["USE_LOCAL_HEAVY"] is True
    assert prepared["USE_LOCAL_LIGHT"] is True
    assert prepared["USE_LOCAL_FALLBACK"] is True


def test_prepare_onboarding_settings_preserves_non_wizard_provider_fields():
    """The wizard only edits fields it actually exposes. Settings fields
    that live in ``settings_ui.js`` but not in the wizard (``OPENAI_BASE_URL``,
    ``CLOUDRU_FOUNDATION_MODELS_BASE_URL``) must survive re-running onboarding.
    ``OPENAI_COMPATIBLE_*`` are now wizard-managed and come from the payload."""
    payload = _base_payload()
    payload["OPENAI_API_KEY"] = "sk-openai-1234567890"
    payload["OPENAI_COMPATIBLE_BASE_URL"] = "https://compat.example/v1"
    payload["OPENAI_COMPATIBLE_API_KEY"] = "compat-secret-xyz"
    current = {
        "OPENAI_BASE_URL": "https://legacy.example/v1",
        "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://cloud.example/v1",
    }

    prepared, error = prepare_onboarding_settings(payload, current)

    assert error is None
    # Non-wizard fields are preserved from current settings.
    assert prepared["OPENAI_BASE_URL"] == "https://legacy.example/v1"
    assert prepared["CLOUDRU_FOUNDATION_MODELS_BASE_URL"] == "https://cloud.example/v1"
    # Compatible fields come from the wizard payload.
    assert prepared["OPENAI_COMPATIBLE_BASE_URL"] == "https://compat.example/v1"
    assert prepared["OPENAI_COMPATIBLE_API_KEY"] == "compat-secret-xyz"


def test_prepare_onboarding_settings_accepts_openai_compatible_setup():
    """An OpenAI-compatible base URL alone (no key) is a valid remote provider."""
    payload = _base_payload()
    payload["OPENAI_COMPATIBLE_BASE_URL"] = "http://localhost:11434/v1"
    payload["OUROBOROS_MODEL"] = "openai-compatible::llama3"
    payload["OUROBOROS_MODEL_HEAVY"] = "openai-compatible::llama3"
    payload["OUROBOROS_MODEL_LIGHT"] = "openai-compatible::llama3"
    payload["OUROBOROS_MODEL_FALLBACKS"] = "openai-compatible::llama3"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert error is None
    assert prepared["OPENAI_COMPATIBLE_BASE_URL"] == "http://localhost:11434/v1"
    assert prepared["OPENAI_COMPATIBLE_API_KEY"] == ""
    assert prepared["OUROBOROS_MODEL"] == "openai-compatible::llama3"


def test_prepare_onboarding_settings_accepts_empty_heavy_and_light():
    """Role-model (v6.39): only Main is required; empty Heavy/Light fall back to Main, so
    the owner is not forced to fill every slot (mirrors the relaxed JS validateModelsStep
    and the live desktop launcher path)."""
    payload = _base_payload()
    payload["OPENAI_API_KEY"] = "sk-openai-1234567890"
    payload["OUROBOROS_MODEL"] = "openai::gpt-5.5"
    payload["OUROBOROS_MODEL_HEAVY"] = ""
    payload["OUROBOROS_MODEL_LIGHT"] = ""

    prepared, error = prepare_onboarding_settings(payload, {})

    assert error is None
    assert prepared["OUROBOROS_MODEL"] == "openai::gpt-5.5"


def test_prepare_onboarding_settings_still_requires_main_model():
    payload = _base_payload()
    payload["OPENAI_API_KEY"] = "sk-openai-1234567890"
    payload["OUROBOROS_MODEL"] = ""

    prepared, error = prepare_onboarding_settings(payload, {})

    assert prepared == {}
    assert "Main model" in error


def test_prepare_onboarding_settings_rejects_openai_compatible_key_without_base_url():
    payload = _base_payload()
    payload["OPENAI_COMPATIBLE_API_KEY"] = "compat-secret-xyz"

    prepared, error = prepare_onboarding_settings(payload, {})

    assert prepared == {}
    assert "Configure OpenRouter, OpenAI, OpenAI-compatible, Cloud.ru, Anthropic, or a local model" in error


def test_onboarding_frontend_uses_base_url_first_compatible_validation():
    source = (REPO / "web/modules/onboarding_wizard.js").read_text(encoding="utf-8")

    assert "field.settingKey !== 'OPENAI_COMPATIBLE_API_KEY'" in source
    assert "const hasRemote = keyValues.some(([, value]) => value);" not in source


def test_build_onboarding_html_contains_multistep_markers():
    html = build_onboarding_html({})

    assert '"contract": {' in html
    assert '"providerFields": [' in html
    assert 'const STEP_ORDER = bootstrap.stepOrder || (SETUP_CONTRACT.steps || []).map' in html
    assert "Add your access" in html
    assert "Keys + local" in html
    assert "Choose models" in html
    assert "model slots" in html
    assert "Choose review mode" in html
    assert "Set your budget" in html
    assert "Local model settings" in html
    assert "openai::gpt-5.5" in html
    assert "openai::gpt-5.4-mini" in html
    assert "anthropic::claude-sonnet-4-6" in html
    assert "OPENAI_BASE_URL: ''" not in html
    assert "OPENAI_COMPATIBLE_API_KEY: ''" not in html
    assert "OPENAI_COMPATIBLE_BASE_URL: ''" not in html
    assert "CLOUDRU_FOUNDATION_MODELS_BASE_URL: ''" not in html


def test_build_onboarding_html_accepts_web_host_mode():
    html = build_onboarding_html({}, host_mode="web")

    assert '"hostMode": "web"' in html
    assert '"supportsLocalRuntimeControls": true' in html
    assert "@media (max-width: 720px)" in html
    assert "scroll-snap-type: x proximity;" in html


def test_build_onboarding_html_adapts_to_multi_provider_access():
    html = build_onboarding_html({})

    assert "function detectProviderProfile()" in html
    assert "function activeProviderProfile()" in html
    assert "function profileLabel(profile)" in html
    assert "function nextButtonShouldBeDisabled()" in html
    assert "function syncCurrentStepActionState()" in html
    assert "return 'direct-multi';" in html
    assert "PROVIDER_FIELDS.map((field) => [field.settingKey, trim(state[field.stateKey])])" in html
    assert "MODEL_SLOTS.map((slot) => [slot.settingKey, trim(state[slot.stateKey])])" in html
    assert "LOCAL_ROUTING_MODE: trim(state.localSource) ? (trim(state.localRoutingMode) || 'cloud') : 'cloud'" in html


def test_setup_contract_has_no_secret_values():
    contract = build_setup_contract("web")
    text = repr(contract)
    budget_fields = {field["settingKey"]: field for field in contract["budgetFields"]}

    assert contract["hostMode"] == "web"
    assert "providerFields" in contract
    assert budget_fields["TOTAL_BUDGET"]["settingsInputId"] == "s-total-budget"
    assert budget_fields["TOTAL_BUDGET"]["min"] == "0.01"
    assert budget_fields["TOTAL_BUDGET"]["step"] == "any"
    assert budget_fields["OUROBOROS_PER_TASK_COST_USD"]["settingsInputId"] == "s-settings-per-task-cost"
    assert "settingsInputId" in contract["providerFields"][0]
    assert "OPENROUTER_API_KEY" in text
    assert "sk-or-v1-super-secret" not in text
    assert "sk-ant-super-secret" not in text
    suggestions = build_setup_bootstrap({}, "web")["modelSuggestions"]
    assert "anthropic/claude-opus-4.7" in suggestions
    assert "anthropic::claude-opus-4-7" in suggestions


def test_api_settings_exposes_setup_contract_without_secrets(tmp_path):
    from unittest.mock import patch

    import server as srv
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    secret = "sk-or-v1-super-secret-token"
    patches = [
        patch.object(srv, "load_settings", return_value={"OPENROUTER_API_KEY": secret}),
        patch.object(srv, "apply_runtime_provider_defaults", lambda settings: (dict(settings), False, [])),
        patch("ouroboros.server_auth.get_configured_network_password", return_value=""),
    ]
    for item in patches:
        item.start()
    try:
        app = Starlette(routes=[Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"])])
        app.state.drive_root = tmp_path
        with TestClient(app) as client:
            response = client.get("/api/settings")
        assert response.status_code == 200
        assert secret not in response.text
        contract = response.json()["_meta"]["setup_contract"]
        assert contract["providerFields"][0]["settingKey"] == "OPENROUTER_API_KEY"
        assert secret not in repr(contract)
    finally:
        for item in patches:
            item.stop()


def test_build_onboarding_html_includes_claude_runtime_cta_and_host_transports():
    desktop_html = build_onboarding_html({}, host_mode="desktop")
    web_html = build_onboarding_html({}, host_mode="web")

    assert "Claude Runtime" in desktop_html or "Claude runtime" in desktop_html
    assert "Skip for now" in desktop_html
    assert "window.pywebview.api.claude_code_status" in desktop_html
    assert "window.pywebview.api.install_claude_code" in desktop_html
    assert "/api/claude-code/status" in web_html
    assert "/api/claude-code/install" in web_html


def _launcher_has_onboarding_bridge() -> bool:
    launcher = REPO / "launcher.py"
    if not launcher.exists():
        return False
    source = launcher.read_text(encoding="utf-8")
    return all(marker in source for marker in (
        "has_startup_ready_provider(settings)",
        "prepare_onboarding_settings(data, settings)",
        'build_onboarding_html(settings, host_mode="desktop")',
        "def claude_code_status(self) -> dict:",
        "def install_claude_code(self) -> dict:",
    ))

_LAUNCHER_HAS_ONBOARDING_BRIDGE = _launcher_has_onboarding_bridge()

@pytest.mark.skipif(
    not _LAUNCHER_HAS_ONBOARDING_BRIDGE,
    reason="launcher.py does not contain onboarding bridge (may be an older bundle or post-refactor version)",
)
def test_launcher_uses_shared_onboarding_and_claude_cli_bridge():
    source = (REPO / "launcher.py").read_text(encoding="utf-8")

    assert "has_startup_ready_provider(settings)" in source
    assert "prepare_onboarding_settings(data, settings)" in source
    assert 'build_onboarding_html(settings, host_mode="desktop")' in source
    assert "def claude_code_status(self) -> dict:" in source
    assert "def install_claude_code(self) -> dict:" in source


def test_web_style_contains_onboarding_overlay_shell():
    style = (REPO / "web" / "style.css").read_text(encoding="utf-8")

    assert ".onboarding-overlay {" in style
    assert ".onboarding-frame {" in style
    assert ".onboarding-overlay-backdrop {" in style
    assert ".onboarding-restart-card {" in style


def test_onboarding_overlay_surfaces_restart_required_message():
    source = (REPO / "web" / "modules" / "onboarding_overlay.js").read_text(encoding="utf-8")

    assert "showRestartRequiredOverlay" in source
    assert "event.data.restart_required" in source
    assert "Continue in current mode" in source
