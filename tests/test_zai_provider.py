"""Z.ai (GLM) direct provider: registry, plan-selected endpoint, the effort
projection at the send boundary, and the 429/1113 billing classification.

Facts pinned here come from the contributor's live probe (PR #1207, 2026-09-21,
Coding Plan key, glm-5.3) and docs.z.ai: the provider accepts exactly
``low``/``high``/``max``, an ABSENT ``reasoning_effort`` is served at max,
thinking cannot be disabled (HTTP 400 code 1210), forced tool_choice works with
thinking on, and plan exhaustion arrives as HTTP 429 code 1113.
"""
import os

import pytest

from ouroboros import provider_models
from ouroboros.llm import LLMClient
from ouroboros.provider_models import (
    DIRECT_PROVIDER_DEFAULTS,
    DIRECT_PROVIDER_REVIEW_ROLES,
    DIRECT_PROVIDER_SCOPE_DEFAULTS,
    ZAI_DIRECT_DEFAULTS,
    ZAI_PLAN_ENDPOINTS,
    ZAI_REASONING_EFFORT_ALIASES,
    migrate_model_value,
    normalize_model_identity,
    normalize_zai_reasoning_effort,
    provider_for_model,
    provider_has_credentials,
    resolve_zai_base_url,
)

_PROVIDER_ENV_KEYS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
    "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
    "ZAI_API_KEY", "ZAI_PLAN",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS",
    "GIGACHAT_USER", "GIGACHAT_PASSWORD", "USE_LOCAL_MAIN",
)


def _clear_provider_env(monkeypatch):
    for key in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _zai_target(model="glm-5.3"):
    return {
        "provider": "zai",
        "resolved_model": model,
        "usage_model": f"zai/{model}",
        "api_key": "sk-x",
        "base_url": ZAI_PLAN_ENDPOINTS["payg"],
        "supports_openrouter_extensions": False,
    }


def _build(target, effort, tool_choice="auto", tools=None):
    client = LLMClient()
    kwargs = client._build_remote_kwargs(
        target, [{"role": "user", "content": "hi"}], effort, 256, tool_choice, None, tools,
    )
    return kwargs, client._pop_effort_clamp_disclosure()


class TestRegistry:
    def test_prefix_routes_direct(self):
        assert provider_for_model("zai::glm-5.3") == "zai"

    def test_slash_form_stays_openrouter(self):
        from ouroboros.pricing import infer_api_key_type

        assert provider_for_model("zai/glm-5.3") == "openrouter"
        assert infer_api_key_type("zai/glm-5.3") == "openrouter"
        assert infer_api_key_type("zai::glm-5.3") == "zai"

    def test_credentials_mapping(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        assert provider_has_credentials("zai") is False
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        assert provider_has_credentials("zai") is True

    def test_direct_defaults_registered(self):
        assert DIRECT_PROVIDER_DEFAULTS["zai"] is ZAI_DIRECT_DEFAULTS
        assert ZAI_DIRECT_DEFAULTS["main"] == "zai::glm-5.3"
        assert ZAI_DIRECT_DEFAULTS["light"] == "zai::glm-5.3-flash"
        assert DIRECT_PROVIDER_REVIEW_ROLES["zai"] == ("main", "main", "main")
        assert DIRECT_PROVIDER_SCOPE_DEFAULTS["zai"] == "zai::glm-5.3"

    def test_migrate_and_normalize_round_trip(self):
        assert migrate_model_value("zai", "zai/glm-5.3") == "zai::glm-5.3"
        assert migrate_model_value("zai", "zai::glm-5.3") == "zai::glm-5.3"
        assert normalize_model_identity("zai::glm-5.3") == "zai/glm-5.3"


class TestPlanSwitch:
    @pytest.mark.parametrize("plan", [None, "", "payg", " PAYG ", "unknown-plan"])
    def test_payg_is_the_default_and_the_fallback(self, plan):
        assert resolve_zai_base_url(plan) == ZAI_PLAN_ENDPOINTS["payg"]

    def test_coding_plan_endpoint(self):
        assert resolve_zai_base_url("coding") == ZAI_PLAN_ENDPOINTS["coding"]
        assert ZAI_PLAN_ENDPOINTS["payg"].startswith("https://api.z.ai/")
        assert ZAI_PLAN_ENDPOINTS["coding"].startswith("https://api.z.ai/")

    def test_resolve_target_uses_plan(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        monkeypatch.setenv("ZAI_PLAN", "coding")
        monkeypatch.setattr(
            "ouroboros.llm_routing.runtime_setting",
            lambda key, default="": os.environ.get(key, default),
        )
        target = LLMClient()._resolve_remote_target("zai::glm-5.3")
        assert target["provider"] == "zai"
        assert target["base_url"] == ZAI_PLAN_ENDPOINTS["coding"]
        assert target["api_key"] == "sk-x"
        assert target["usage_model"] == "zai/glm-5.3"

    def test_route_readers_follow_the_plan(self, monkeypatch):
        # The Capability Evidence route identity (main route + reviewer route)
        # must name the plan's endpoint, exactly as MiniMax's follows its region.
        from ouroboros.gateway.settings import _active_main_route
        from ouroboros.reviewer_window import reviewer_route

        monkeypatch.setattr("ouroboros.config.runtime_settings", lambda: {"ZAI_PLAN": "coding"})
        assert reviewer_route("zai::glm-5.3") == ("zai", ZAI_PLAN_ENDPOINTS["coding"])
        route = _active_main_route({"OUROBOROS_MODEL": "zai::glm-5.3", "ZAI_PLAN": "coding"})
        assert (route["provider"], route["base_url"]) == ("zai", ZAI_PLAN_ENDPOINTS["coding"])
        assert _active_main_route({"OUROBOROS_MODEL": "zai::glm-5.3"})["base_url"] == ZAI_PLAN_ENDPOINTS["payg"]

    def test_provider_test_rejects_an_unknown_plan(self, monkeypatch):
        from ouroboros.gateway import models as provider_api

        monkeypatch.setattr(provider_api, "load_settings", lambda: {})
        monkeypatch.setattr(
            provider_api, "_run_provider_test_with_settings",
            lambda *_args: (_ for _ in ()).throw(AssertionError("must not probe")),
        )
        body = provider_api._run_provider_test("zai", {"ZAI_API_KEY": "x", "ZAI_PLAN": "codign"})
        assert body == {"error": "unknown Z.ai plan", "_http_status": 400}

    def test_plan_alone_is_not_a_provider(self):
        # A plan is a transport choice, not a credential: a draft carrying only
        # ZAI_PLAN must be refused exactly like a MiniMax region without a key,
        # while the same draft with the key is accepted.
        from ouroboros.settings_setup_contract import validate_setup_payload

        plan_only = {"ZAI_PLAN": "coding", "OUROBOROS_MODEL": "zai::glm-5.3",
                     "OUROBOROS_MODEL_LIGHT": "zai::glm-5.3-flash", "OUROBOROS_MODEL_FALLBACKS": "zai::glm-5.3-flash"}
        _prepared, error = validate_setup_payload(plan_only, {})
        assert error
        _prepared, error = validate_setup_payload({**plan_only, "ZAI_API_KEY": "sk-zai-key-1234567890"}, {})
        assert not error


class TestEffortCarriage:
    """The canonical scale is always projected onto Z.ai's low/high/max enum:
    an absent tier would be served (and billed) at max."""

    @pytest.mark.parametrize(
        ("requested", "wire"),
        [
            ("none", "low"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
            ("ultra", "max"),
        ],
    )
    def test_projection_reaches_the_wire(self, requested, wire):
        kwargs, note = _build(_zai_target(), requested)
        assert kwargs["reasoning_effort"] == wire
        assert "thinking" not in (kwargs.get("extra_body") or {})
        if requested == wire:
            assert note is None
        else:
            assert note == {
                "requested": requested, "applied": wire,
                "reason": "provider_wire_mapping", "model": "glm-5.3",
            }

    def test_table_stays_inside_the_provider_enum(self):
        assert set(ZAI_REASONING_EFFORT_ALIASES.values()) == {"low", "high", "max"}
        assert normalize_zai_reasoning_effort("not-a-tier") == "low"

    def test_forced_tool_choice_keeps_thinking_on(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        kwargs, _ = _build(_zai_target(), "high", tool_choice="required", tools=tools)
        assert kwargs["reasoning_effort"] == "high"
        assert "thinking" not in (kwargs.get("extra_body") or {})

    def test_generic_compatible_lane_is_untouched(self):
        # The same GLM model id on an owner's OpenAI-compatible endpoint keeps
        # today's behavior: the projection is keyed on the zai provider id,
        # never on the model name.
        target = {
            "provider": "openai-compatible", "resolved_model": "glm-5.3",
            "usage_model": "openai-compatible::glm-5.3", "api_key": "",
            "base_url": "http://127.0.0.1:11434/v1", "supports_openrouter_extensions": False,
        }
        kwargs, note = _build(target, "medium")
        assert "reasoning_effort" not in kwargs
        assert note is None


class TestSingleProviderIndependence:
    def test_exclusive_direct_env_detection(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        from ouroboros.config import _exclusive_direct_remote_provider_env

        assert _exclusive_direct_remote_provider_env() == "zai"

    def test_startup_gate_accepts_zai_only(self):
        from ouroboros.server_runtime import (
            _exclusive_direct_remote_provider,
            has_remote_provider,
            has_startup_ready_provider,
        )

        settings = {"ZAI_API_KEY": "sk-x"}
        assert has_remote_provider(settings) is True
        assert has_startup_ready_provider(settings) is True
        assert _exclusive_direct_remote_provider(settings) == "zai"

    def test_review_fallback_compiles_for_zai(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        monkeypatch.setenv("OUROBOROS_MODEL", "zai::glm-5.3")
        monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "zai::glm-5.3-flash")
        monkeypatch.setattr(
            "ouroboros.review_model_routes.runtime_setting",
            lambda key, default="": os.environ.get(key, default),
        )
        from ouroboros.config import get_review_models

        assert get_review_models() == ["zai::glm-5.3"] * 3

    def test_local_only_review_route_sees_zai(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("USE_LOCAL_MAIN", "1")
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        assert provider_models.local_only_review_route_env() is False


class TestSecretSurfaces:
    def test_forbidden_for_skills_and_masked(self):
        from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
        from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS

        assert "ZAI_API_KEY" in FORBIDDEN_SKILL_SETTINGS
        assert "ZAI_API_KEY" in MASKED_SECRET_SETTING_KEYS

    def test_settings_defaults(self):
        from ouroboros.config import SETTINGS_DEFAULTS

        assert SETTINGS_DEFAULTS["ZAI_API_KEY"] == ""
        assert SETTINGS_DEFAULTS["ZAI_PLAN"] == ""


class TestSafetyRouting:
    def test_zai_only_install_reaches_the_real_safety_check(self, monkeypatch):
        """A zai-only install must reach the remote safety check, not fail open."""
        from ouroboros import safety

        _clear_provider_env(monkeypatch)
        assert safety._any_remote_provider_configured() is False
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        assert safety._any_remote_provider_configured() is True
        assert safety._PROVIDER_KEY_ENV["zai"] == "ZAI_API_KEY"

    def test_light_model_reaches_its_provider_key(self, monkeypatch):
        from ouroboros import safety
        from ouroboros.pricing import infer_api_key_type

        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("ZAI_API_KEY", "sk-x")
        assert safety._PROVIDER_KEY_ENV.get(infer_api_key_type("zai::glm-5.3-flash")) == "ZAI_API_KEY"


class TestProbeBilling:
    """HTTP 429 code 1113 "Insufficient balance" is billing, not rate limiting."""

    def test_1113_maps_to_no_credits(self):
        from ouroboros.llm_probe import controlled_probe_error

        class Exhausted(Exception):
            status_code = 429
            code = "1113"
            type = ""

        result = controlled_probe_error(Exhausted("Insufficient balance"))
        assert result["error"] == "No credits"
        assert result["status_code"] == 429

    def test_task_loop_does_not_retry_an_exhausted_plan(self):
        from ouroboros.loop_llm_call import classify_llm_exception

        class Exhausted(Exception):
            status_code = 429
            code = "1113"
            body = {"error": {"code": "1113", "message": "Insufficient balance"}}

        exhausted = classify_llm_exception(Exhausted("Insufficient balance"))
        assert exhausted.kind == "quota_exhausted"
        assert exhausted.retry_same_request is False
        # An ordinary 429 keeps its transient, retryable classification.
        assert classify_llm_exception(RuntimeError("Error code: 429 - too many requests")).retry_same_request is True

    def test_plain_429_stays_rate_limited(self):
        from ouroboros.llm_probe import controlled_probe_error

        class Plain(Exception):
            status_code = 429
            code = ""
            type = ""

        assert controlled_probe_error(Plain("too many requests"))["error"] == "Rate limited"
