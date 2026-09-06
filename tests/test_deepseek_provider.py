"""Direct DeepSeek provider regressions (v4 family, OpenAI-compatible endpoint).

Covers the single-provider independence contract (DEVELOPMENT.md "Provider
Independence") and the two DeepSeek-specific wire classes established by live
probes (2026-09-01):

- the effort-carrying route: DeepSeek's Chat API takes ``low``/``high``/``max``
  (``medium``/``xhigh`` are aliases of ``high``) and switches thinking off via
  ``thinking.type=disabled``, so the lane projects the canonical scale onto that
  dialect instead of dropping effort like other generic compatible lanes;
- the reasoning-echo REQUIREMENT: tool-bearing requests must pass every
  assistant turn's ``reasoning_content`` back (v4-pro enforces with a 400;
  an explicit empty string satisfies the gate for turns produced elsewhere).
"""

import pytest

from ouroboros import provider_models
from ouroboros.llm import LLMClient
from ouroboros.provider_models import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DIRECT_DEFAULTS,
    DIRECT_PROVIDER_DEFAULTS,
    DIRECT_PROVIDER_REVIEW_ROLES,
    DIRECT_PROVIDER_SCOPE_DEFAULTS,
    migrate_model_value,
    normalize_model_identity,
    provider_for_model,
    provider_has_credentials,
    supports_vision,
)
from ouroboros.request_wire_contract import payload_effort, reasoning_carrier
from ouroboros.request_wire_recovery import (
    prepare_wire_payload_for_send,
    request_wire_call_scope,
)


def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_BASE_URL",
        "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
        "CLOUDRU_FOUNDATION_MODELS_API_KEY", "GIGACHAT_CREDENTIALS",
        "GIGACHAT_USER", "GIGACHAT_PASSWORD", "USE_LOCAL_MAIN",
    ):
        monkeypatch.delenv(key, raising=False)


class TestRegistry:
    def test_prefix_routes_direct(self):
        assert provider_for_model("deepseek::deepseek-v4-pro") == "deepseek"

    def test_slash_form_stays_openrouter(self):
        # deepseek/ is a REAL OpenRouter vendor namespace (the CI canary uses
        # it); only the :: prefix selects the direct route.
        assert provider_for_model("deepseek/deepseek-v4-pro") == "openrouter"
        from ouroboros.pricing import infer_api_key_type
        assert infer_api_key_type("deepseek/deepseek-v4-pro") == "openrouter"
        assert infer_api_key_type("deepseek::deepseek-v4-pro") == "deepseek"

    def test_credentials_mapping(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        assert provider_has_credentials("deepseek") is False
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        assert provider_has_credentials("deepseek") is True
        assert provider_models.model_has_credentials("deepseek::deepseek-v4-flash") is True

    def test_direct_defaults_registered(self):
        assert DIRECT_PROVIDER_DEFAULTS["deepseek"] is DEEPSEEK_DIRECT_DEFAULTS
        assert DEEPSEEK_DIRECT_DEFAULTS["main"] == "deepseek::deepseek-v4-pro"
        assert DEEPSEEK_DIRECT_DEFAULTS["light"] == "deepseek::deepseek-v4-flash"
        assert DEEPSEEK_DIRECT_DEFAULTS["deep_self_review"] == "deepseek::deepseek-v4-pro"
        assert DIRECT_PROVIDER_REVIEW_ROLES["deepseek"] == ("main", "main", "main")
        assert DIRECT_PROVIDER_SCOPE_DEFAULTS["deepseek"] == "deepseek::deepseek-v4-pro"

    def test_migrate_and_normalize_round_trip(self):
        assert migrate_model_value("deepseek", "deepseek/deepseek-v4-pro") == "deepseek::deepseek-v4-pro"
        assert migrate_model_value("deepseek", "deepseek::deepseek-v4-pro") == "deepseek::deepseek-v4-pro"
        assert normalize_model_identity("deepseek::deepseek-v4-flash") == "deepseek/deepseek-v4-flash"

    def test_vision_narrow_prefix(self):
        assert supports_vision("deepseek::deepseek-v4-flash-vision-exp") is True
        assert supports_vision("deepseek::deepseek-v4-flash") is False
        assert supports_vision("deepseek/deepseek-chat") is False


class TestSingleProviderIndependence:
    def test_exclusive_direct_env_detection(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        from ouroboros.config import _exclusive_direct_remote_provider_env
        assert _exclusive_direct_remote_provider_env() == "deepseek"

    def test_review_and_scope_fallback_compile(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        monkeypatch.setenv("OUROBOROS_MODEL", "deepseek::deepseek-v4-pro")
        monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "deepseek::deepseek-v4-flash")
        from ouroboros.config import get_review_models, get_scope_review_models
        assert get_review_models() == ["deepseek::deepseek-v4-pro"] * 3
        assert get_scope_review_models() == ["deepseek::deepseek-v4-pro"]

    def test_startup_gate_accepts_deepseek_only(self):
        from ouroboros.server_runtime import (
            _exclusive_direct_remote_provider,
            has_remote_provider,
            has_startup_ready_provider,
        )
        settings = {"DEEPSEEK_API_KEY": "sk-x"}
        assert has_remote_provider(settings) is True
        assert has_startup_ready_provider(settings) is True
        assert _exclusive_direct_remote_provider(settings) == "deepseek"

    def test_local_only_review_route_sees_deepseek(self, monkeypatch):
        _clear_provider_env(monkeypatch)
        monkeypatch.setenv("USE_LOCAL_MAIN", "1")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        # A live remote DeepSeek credential means review slots must NOT be
        # forced onto the local route.
        assert provider_models.local_only_review_route_env() is False

    def test_secret_surfaces_cover_deepseek(self):
        from ouroboros.contracts.plugin_api import FORBIDDEN_SKILL_SETTINGS
        from ouroboros.secret_masking import MASKED_SECRET_SETTING_KEYS
        assert "DEEPSEEK_API_KEY" in FORBIDDEN_SKILL_SETTINGS
        assert "DEEPSEEK_API_KEY" in MASKED_SECRET_SETTING_KEYS
        from ouroboros.config import SETTINGS_DEFAULTS
        assert SETTINGS_DEFAULTS["DEEPSEEK_API_KEY"] == ""


class TestWireProjection:
    def _target(self, monkeypatch, model="deepseek::deepseek-v4-flash"):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        return LLMClient()._resolve_remote_target(model)

    def test_resolve_remote_target_shape(self, monkeypatch):
        target = self._target(monkeypatch)
        assert target["provider"] == "deepseek"
        assert target["base_url"] == DEEPSEEK_BASE_URL
        assert target["api_key"] == "sk-x"
        assert target["usage_model"] == "deepseek/deepseek-v4-flash"
        assert target["requires_reasoning_echo"] is True
        assert target["supports_openrouter_extensions"] is False

    def test_effort_carried_with_max_tokens_carrier(self, monkeypatch):
        client = LLMClient()
        target = self._target(monkeypatch)
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], "high", 256, "auto", None, None,
        )
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_tokens"] == 256
        assert "max_completion_tokens" not in kwargs
        assert "extra_body" not in kwargs

    @pytest.mark.parametrize(
        ("requested", "wire"),
        [
            ("none", None),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "max"),
            ("ultra", "max"),
        ],
    )
    def test_effort_projection_matches_deepseek_chat_contract(
        self, monkeypatch, requested, wire,
    ):
        # Official contract (api-docs.deepseek.com, Thinking Mode): the enum is
        # low/high/max, medium/xhigh alias high, and ``none`` is the separate
        # ``thinking.type=disabled`` toggle, never a reasoning_effort value.
        client = LLMClient()
        target = self._target(monkeypatch)
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], requested, 256, "auto", None, None,
        )
        assert kwargs.get("reasoning_effort") == wire
        if wire is None:
            assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        else:
            assert "extra_body" not in kwargs
        note = client._pop_effort_clamp_disclosure()
        if wire is not None and wire != requested:
            assert note == {
                "requested": requested,
                "applied": wire,
                "reason": "provider_wire_mapping",
                "model": "deepseek-v4-flash",
            }
        else:
            assert note is None

    @pytest.mark.parametrize(
        "choice",
        ["required", {"type": "function", "function": {"name": "get_date"}}],
    )
    def test_forced_tool_choice_is_served_without_thinking(self, monkeypatch, choice):
        # Live-probed 2026-09-03 on v4-flash and v4-pro: thinking mode 400s on
        # tool_choice required/named ("Thinking mode does not support this
        # tool_choice"), while auto/none work; every form works with thinking
        # disabled. The caller's structural demand (a tool call WILL come
        # back) wins over reasoning on that one call, and usage says so.
        client = LLMClient()
        target = self._target(monkeypatch)
        tool = {"type": "function", "function": {
            "name": "get_date", "parameters": {"type": "object", "properties": {}}}}
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], "high", 256, choice, None, [tool],
        )
        assert "reasoning_effort" not in kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert kwargs["tool_choice"] == choice
        assert client._pop_effort_clamp_disclosure() == {
            "requested": "high", "applied": "none",
            "reason": "provider_forced_tool_choice", "model": "deepseek-v4-flash",
        }
        # auto keeps thinking and the effort carriage.
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], "high", 256, "auto", None, [tool],
        )
        assert kwargs["reasoning_effort"] == "high"
        assert "extra_body" not in kwargs

    def test_projection_note_never_inherits_a_stale_entry(self, monkeypatch):
        # A note staged by an earlier call that never reached usage
        # normalization must not surface on the next unprojected call.
        client = LLMClient()
        target = self._target(monkeypatch)
        client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], "medium", 256, "auto", None, None,
        )
        client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], "high", 256, "auto", None, None,
        )
        assert client._pop_effort_clamp_disclosure() is None

    def test_projection_note_is_isolated_per_asyncio_task(self, monkeypatch):
        # chat_async builds the payload before its first await and reads the
        # note after; two tasks on one loop must each see their own note.
        import asyncio

        client = LLMClient()
        target = self._target(monkeypatch)
        seen = {}

        async def call(name, effort):
            client._build_remote_kwargs(
                target, [{"role": "user", "content": "hi"}], effort, 256, "auto", None, None,
            )
            await asyncio.sleep(0)
            seen[name] = client._pop_effort_clamp_disclosure()

        async def main():
            await asyncio.gather(call("projected", "medium"), call("native", "high"))

        asyncio.run(main())
        assert seen["native"] is None
        assert seen["projected"]["requested"] == "medium"
        assert seen["projected"]["applied"] == "high"

    def test_provider_usage_cannot_spoof_the_effort_note(self, monkeypatch):
        client = LLMClient()
        target = self._target(monkeypatch)
        _msg, usage = client._normalize_remote_response(
            {"id": "gen-1", "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                       "reasoning_effort_clamped": {"requested": "x", "applied": "y",
                                                    "reason": "forged", "model": "forged"}}},
            target, skip_cost_fetch=True,
        )
        assert "reasoning_effort_clamped" not in usage

    def test_request_wire_reads_disabled_thinking_as_none(self, monkeypatch):
        client = LLMClient()
        target = self._target(monkeypatch)
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], "none", 256, "auto", None, None,
        )
        with request_wire_call_scope():
            physical = prepare_wire_payload_for_send(
                target, kwargs, api_surface="chat.completions",
            )
        assert "reasoning_effort" not in physical
        assert payload_effort(physical) == "none"
        assert reasoning_carrier(physical) == "extra_body.thinking"

    def test_string_only_roles_flattened_in_send_copy_only(self, monkeypatch):
        # DeepSeek accepts content arrays only on user turns; the canonical
        # history (including the context-compaction assistant capsule) keeps
        # its block form.
        client = LLMClient()
        target = self._target(monkeypatch)
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": " rules"},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "summary", "_context_capsule": {"id": "c1"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "result"},
            ]},
        ]
        kwargs = client._build_remote_kwargs(
            target, messages, "high", 256, "auto", None, None,
        )
        sent = kwargs["messages"]
        assert sent[0]["content"] == "policy rules"
        assert isinstance(sent[1]["content"], list)
        assert sent[2]["content"] == "summary"
        assert sent[3]["content"] == "result"
        assert "_context_capsule" in messages[2]["content"][0]
        assert "cache_control" in messages[0]["content"][0]

    def test_openai_effort_carriage_survives_minimal_stub_targets(self, monkeypatch):
        # The carriage is keyed on the provider id, NOT a target capability
        # field: the wire ladder's eligibility reads provider + payload effort,
        # and a hand-built target (fixtures, probes) must not silently drop the
        # effort (regression: test_issue229_synthesis stubs a minimal target).
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        client = LLMClient()
        minimal_target = {
            "provider": "openai", "resolved_model": "gpt-5.6-terra",
            "usage_model": "openai/gpt-5.6-terra", "api_key": "k",
            "base_url": "https://api.openai.com/v1", "default_headers": {},
            "supports_openrouter_extensions": False,
            "supports_generation_cost": False,
        }
        kwargs = client._build_remote_kwargs(
            minimal_target, [{"role": "user", "content": "hi"}], "high", 128, "auto", None, None,
        )
        assert kwargs["reasoning_effort"] == "high"

    def test_reasoning_content_replayed_and_gap_filled(self, monkeypatch):
        client = LLMClient()
        target = self._target(monkeypatch)
        messages = [
            {"role": "user", "content": "weather?"},
            {  # DeepSeek's own turn: reasoning must ride back verbatim.
                "role": "assistant", "content": "",
                "reasoning_content": "call the tool",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "t", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "18C"},
            {  # Foreign-model turn (no reasoning): the strict v4-pro gate
                # still demands the field — an explicit "" satisfies it.
                "role": "assistant", "content": "done",
            },
        ]
        kwargs = client._build_remote_kwargs(
            target, messages, "high", 256, "auto", None,
            [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}],
        )
        sent = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
        assert sent[0]["reasoning_content"] == "call the tool"
        assert sent[1]["reasoning_content"] == ""

    def test_other_compatible_lanes_still_strip(self, monkeypatch):
        monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "k")
        client = LLMClient()
        target = client._resolve_remote_target("cloudru::zai-org/GLM-4.7")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": "glm echo"},
        ]
        kwargs = client._build_remote_kwargs(
            target, messages, "high", 256, "auto", None, None,
        )
        sent = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
        assert "reasoning_content" not in sent[0]
        assert "reasoning_effort" not in kwargs  # non-carrying lane unchanged

    def test_normalize_keeps_deepseek_reasoning_in_transcript(self):
        client = LLMClient(api_key="x")
        resp = {
            "id": "1",
            "choices": [{"message": {
                "role": "assistant", "content": "4",
                "reasoning_content": "2+2 -> 4",
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        target = {
            "provider": "deepseek",
            "usage_model": "deepseek/deepseek-v4-flash",
            "supports_openrouter_extensions": False,
            "supports_generation_cost": False,
        }
        msg, _usage = client._normalize_remote_response(resp, target, skip_cost_fetch=True)
        assert msg["reasoning_content"] == "2+2 -> 4"

    def test_normalize_cached_tokens_top_level_fallback(self):
        client = LLMClient(api_key="x")
        resp = {
            "id": "1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            # DeepSeek-shaped usage: top-level hit/miss beside an EMPTY
            # details block — the fallback must still account the cache hit.
            "usage": {
                "prompt_tokens": 446, "completion_tokens": 19,
                "prompt_tokens_details": {},
                "prompt_cache_hit_tokens": 384, "prompt_cache_miss_tokens": 62,
            },
        }
        target = {
            "provider": "deepseek",
            "usage_model": "deepseek/deepseek-v4-flash",
            "supports_openrouter_extensions": False,
            "supports_generation_cost": False,
        }
        _msg, usage = client._normalize_remote_response(resp, target, skip_cost_fetch=True)
        assert usage["cached_tokens"] == 384

    def test_cross_family_switch_strips_deepseek_reasoning(self):
        messages = [
            {"role": "assistant", "content": "x", "reasoning_content": "ds"},
        ]
        out = LLMClient.sanitize_reasoning_on_model_switch(
            messages, "deepseek::deepseek-v4-pro", "google/gemini-3.7-flash",
        )
        assert "reasoning_content" not in out[0]
        same = LLMClient.sanitize_reasoning_on_model_switch(
            messages, "deepseek::deepseek-v4-pro", "deepseek::deepseek-v4-flash",
        )
        assert same[0].get("reasoning_content") == "ds"

    def test_vision_images_survive_for_vision_variant_only(self, monkeypatch):
        client = LLMClient()
        image_msg = [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]
        vision_target = self._target(monkeypatch, "deepseek::deepseek-v4-flash-vision-exp")
        kwargs = client._build_remote_kwargs(
            vision_target, image_msg, "high", 128, "auto", None, None,
        )
        blocks = kwargs["messages"][0]["content"]
        assert any(isinstance(b, dict) and b.get("type") == "image_url" for b in blocks)

        blind_target = self._target(monkeypatch, "deepseek::deepseek-v4-flash")
        kwargs = client._build_remote_kwargs(
            blind_target, image_msg, "high", 128, "auto", None, None,
        )
        blocks = kwargs["messages"][0]["content"]
        assert not any(isinstance(b, dict) and b.get("type") == "image_url" for b in blocks)

    def test_openrouter_lane_neither_leaks_nor_pins_on_deepseek_residue(self, monkeypatch):
        # A mixed transcript (direct-DeepSeek turns replayed on an OpenRouter
        # model without passing a switch seam) must NOT forward the
        # deepseek-private reasoning_content to OpenRouter, and must NOT trip
        # the replay-artifact allow_fallbacks=False pin off that residue.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
        client = LLMClient()
        target = client._resolve_remote_target("google/gemini-3.7-flash")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": "ds thoughts"},
            {"role": "user", "content": "again"},
        ]
        kwargs = client._build_remote_kwargs(
            target, messages, "high", 128, "auto", None, None,
        )
        sent = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
        assert all("reasoning_content" not in m for m in sent)
        provider_body = (kwargs.get("extra_body") or {}).get("provider") or {}
        assert provider_body.get("allow_fallbacks") is not False
        # The canonical transcript is untouched.
        assert messages[1]["reasoning_content"] == "ds thoughts"

    def test_openrouter_lane_strips_falsy_reasoning_residue_too(self, monkeypatch):
        # The scrub is keyed on key PRESENCE: an empty-string echo (a legal
        # kept value on the deepseek lane) and a legacy null must both stay
        # off the OpenRouter wire, not just truthy thoughts.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
        client = LLMClient()
        target = client._resolve_remote_target("google/gemini-3.7-flash")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": ""},
            {"role": "assistant", "content": "y", "reasoning_content": None},
        ]
        kwargs = client._build_remote_kwargs(
            target, messages, "high", 128, "auto", None, None,
        )
        sent = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
        assert all("reasoning_content" not in m for m in sent)

    def test_openai_compatible_vendor_form_vision_stays_sighted(self, monkeypatch):
        # The qualified identity (openai-compatible/<id>) can never match the
        # vision prefixes; the BARE vendor-form id legitimately does. The
        # either-identity judgment keeps that lane sighted.
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "k")
        monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1")
        client = LLMClient()
        target = client._resolve_remote_target("openai-compatible::qwen/qwen2.5-vl-7b")
        image_msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]
        kwargs = client._build_remote_kwargs(
            target, image_msg, "high", 128, "auto", None, None,
        )
        blocks = kwargs["messages"][0]["content"]
        assert any(isinstance(b, dict) and b.get("type") == "image_url" for b in blocks)

    def test_direct_openai_vision_judged_on_qualified_identity(self, monkeypatch):
        # Regression for the latent direct-lane class: the bare resolved id
        # never matched the slash-form vision prefixes, so every direct route
        # was captioned/placeholder'd regardless of real capability.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        client = LLMClient()
        target = client._resolve_remote_target("openai::gpt-5.5")
        image_msg = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]
        kwargs = client._build_remote_kwargs(
            target, image_msg, "high", 128, "auto", None, None,
        )
        blocks = kwargs["messages"][0]["content"]
        assert any(isinstance(b, dict) and b.get("type") == "image_url" for b in blocks)


class TestReviewWaveHardening:
    """Phase A review-wave fixes: witness eligibility, null reasoning, fit basis."""

    def test_deepseek_usage_is_density_witness_eligible(self, tmp_path):
        # DeepSeek's automatic cache makes nearly every warm call cache-bearing;
        # excluding it from the cache-inclusive set starved the route of
        # witnesses after the first cold call (probed: prompt_tokens = hit + miss).
        from ouroboros import usage_accounting as ua
        from ouroboros.capability_evidence import _DENSITY_MEMO, get_token_density
        from ouroboros.provider_models import normalize_model_identity

        _DENSITY_MEMO.clear()
        try:
            root = tmp_path / "deepseek"
            ua._observe_token_density(
                ua.AttemptRequest(
                    model="deepseek/deepseek-v4-pro",
                    provider="deepseek",
                    prompt_tokens_estimate=1_000_000,
                    drive_root=root,
                ),
                {"prompt_tokens": 1_500_000, "cached_tokens": 900_000},
            )
            measured = get_token_density(root, normalize_model_identity("deepseek/deepseek-v4-pro"))
            assert abs(measured - 1.5) < 1e-6
        finally:
            # The memo is process-global and keyed WITHOUT drive_root: leaving
            # this tmp_path measurement behind would poison a co-located test
            # reading density for the same model id.
            _DENSITY_MEMO.clear()

    def test_normalize_drops_non_string_reasoning_content(self):
        # A server-emitted null would live on the canonical assistant turn and
        # replay as JSON null against a gate probed only for strings, with no
        # message-level 400 recovery on the direct lane.
        client = LLMClient(api_key="x")
        resp = {
            "id": "1",
            "choices": [{"message": {
                "role": "assistant", "content": "4",
                "reasoning_content": None,
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        target = {
            "provider": "deepseek",
            "usage_model": "deepseek/deepseek-v4-flash",
            "supports_openrouter_extensions": False,
            "supports_generation_cost": False,
        }
        msg, _usage = client._normalize_remote_response(resp, target, skip_cost_fetch=True)
        assert "reasoning_content" not in msg

    def test_echo_fill_coerces_non_string_reasoning(self, monkeypatch):
        # An imported/legacy transcript can already carry a null; setdefault
        # cannot repair an existing key, so the fill must coerce by type.
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
        client = LLMClient()
        target = client._resolve_remote_target("deepseek::deepseek-v4-pro")
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": None},
        ]
        kwargs = client._build_remote_kwargs(
            target, messages, "high", 256, "auto", None,
            [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}],
        )
        sent = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
        assert sent[0]["reasoning_content"] == ""

    def test_estimate_message_chars_counts_replayed_reasoning(self):
        # Replayed reasoning is real wire prompt on the echo lane: the planning
        # basis must see it or fit drift grows with transcript length.
        from ouroboros.context_budget import estimate_message_chars

        plain = [{"role": "assistant", "content": "ab"}]
        with_reasoning = [{"role": "assistant", "content": "ab", "reasoning_content": "cdef"}]
        assert estimate_message_chars(with_reasoning) == estimate_message_chars(plain) + 4

    def test_remote_fit_estimator_counts_replayed_reasoning(self):
        # The PRODUCTION remote-fit path: estimate_context_prompt_tokens
        # serializes message dicts recursively, so the deepseek lane's kept
        # reasoning_content must grow the estimate — this is the guarantee the
        # fit machinery actually runs on for remote sends.
        from ouroboros.context_fit import estimate_context_prompt_tokens

        plain = [{"role": "assistant", "content": "ab"}]
        with_reasoning = [
            {"role": "assistant", "content": "ab", "reasoning_content": "r" * 4000},
        ]
        assert (
            estimate_context_prompt_tokens(with_reasoning, provider="deepseek")
            > estimate_context_prompt_tokens(plain, provider="deepseek") + 500
        )
