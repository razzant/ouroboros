"""Tests for LLMClient.supported_parameters cache and dynamic kwarg filtering.

v4.33.0 adds a per-process cache of OpenRouter model capabilities so we can
strip sampling parameters (`temperature`, `top_p`, `top_k`) the resolved
model doesn't list in `supported_parameters`. Combined with
`provider.require_parameters: true` on Anthropic-prefixed models, unknown
params used to cause 404 "No endpoints found" from OpenRouter (this is why
`anthropic/claude-opus-4.6` was silently dropped from every triad review
for the whole v4.32.x line — it simply doesn't support `temperature`).

These tests cover:
- A known-incompatible model has `temperature` stripped.
- A known-compatible model keeps `temperature`.
- Fetch failure (network/parse error) falls back to broad support — no stripping.
- The cache is populated once per process, not per call.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _reset_llm_cache():
    """Reset the class-level supported_parameters cache before/after each test."""
    from ouroboros.llm import LLMClient
    LLMClient._SUPPORTED_PARAMS_CACHE.clear()
    LLMClient._SUPPORTED_PARAMS_FETCHED = False
    LLMClient._REJECTED_PARAMS_CACHE.clear()
    yield
    LLMClient._SUPPORTED_PARAMS_CACHE.clear()
    LLMClient._SUPPORTED_PARAMS_FETCHED = False
    LLMClient._REJECTED_PARAMS_CACHE.clear()


def _install_fake_response(monkeypatch, data: dict[str, Any]) -> dict[str, int]:
    """Patch requests.get used by _fetch_openrouter_capabilities with a canned response."""
    call_count = {"n": 0}

    class _Resp:
        status_code = 200
        def json(self):
            return data

    def fake_get(url: str, timeout: int = 15):
        call_count["n"] += 1
        return _Resp()

    # _fetch_openrouter_capabilities does `import requests` lazily inside the function,
    # so we patch the module attribute that it will see after the lazy import.
    import requests as _real_requests
    monkeypatch.setattr(_real_requests, "get", fake_get)
    return call_count


class TestSupportedParametersFilter:
    @pytest.mark.parametrize("model", ["anthropic/claude-fable-5.1", "vendor/future-model"])
    @pytest.mark.parametrize("no_proxy", [False, True])
    @pytest.mark.parametrize("known_metadata", [False, True])
    def test_chat_omits_neutral_choice_before_physical_binding(
        self, monkeypatch, tmp_path, model, no_proxy, known_metadata,
    ):
        """An ordinary tool request must survive a catalog without tool_choice."""
        import copy
        from types import SimpleNamespace

        from ouroboros import request_wire_contract, usage_accounting
        from ouroboros.llm import LLMClient
        from ouroboros.request_wire_recovery import current_wire_candidate
        from ouroboros.usage_accounting import UsageScope, usage_scope

        monkeypatch.setattr(request_wire_contract, "canonical_wire_evidence_root", lambda: tmp_path)
        monkeypatch.setattr(usage_accounting, "estimate_cost_optional", lambda *a, **kw: 0.0)
        monkeypatch.setenv("OUROBOROS_OR_PROVIDER", '{"require_parameters":true,"allow_fallbacks":true}')
        client = LLMClient(api_key="test")
        supported = {"tools", "reasoning", "max_tokens"} if known_metadata else None
        monkeypatch.setattr(client, "_get_supported_parameters", lambda _model: supported)
        # no_proxy deliberately skips a network fetch, but still uses a warm cache.
        monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", known_metadata)
        if known_metadata:
            LLMClient._SUPPORTED_PARAMS_CACHE[model] = supported
        tool = {"type": "function", "function": {
            "name": "probe", "description": "Record a probe.",
            "parameters": {"type": "object", "properties": {}},
        }}
        sent = []

        class ParameterRejection(RuntimeError):
            status_code = 404
            body = {"error": {
                "code": 404,
                "message": "No endpoints found that can handle the requested parameters",
                "metadata": {"failed_routing_step": "Filter by Parameters"},
            }}

        class Response:
            def model_dump(self):
                return {"model": model, "provider": "Test provider", "choices": [{
                    "finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "call-probe", "type": "function",
                                        "function": {"name": "probe", "arguments": "{}"}}],
                    },
                }], "usage": {"prompt_tokens": 11, "completion_tokens": 2, "cost": 0.0}}

        def create(**payload):
            sent.append(copy.deepcopy(payload))
            if "tool_choice" in payload:
                raise ParameterRejection(ParameterRejection.body["error"]["message"])
            candidate = current_wire_candidate()
            assert candidate.source_profile.tool_choice == "omitted"
            assert candidate.accepted_profile.tool_choice == "omitted"
            return Response()

        remote = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(client, "_get_remote_client", lambda _target: remote)
        monkeypatch.setattr(client, "_make_no_proxy_client", lambda _target, timeout=None: (
            remote, SimpleNamespace(close=lambda: None),
        ))
        with usage_scope(UsageScope(drive_root=tmp_path, task_id="neutral-choice")):
            message, usage = client.chat(
                messages=[{"role": "user", "content": "Call probe."}], model=model,
                tools=[tool], reasoning_effort="medium", no_proxy=no_proxy,
            )
        assert len(sent) == 1
        assert sent[0]["model"] == model
        assert sent[0]["tools"][0]["function"] == tool["function"]
        assert sent[0]["extra_body"]["provider"] == {
            "require_parameters": True, "allow_fallbacks": True,
        }
        assert sent[0]["extra_body"]["reasoning"]["effort"] == "medium"
        assert sent[0]["max_tokens"] == 65536
        assert message["tool_calls"][0]["function"]["name"] == "probe"
        assert usage["request_wire"]["applied_actions"] == []
        assert len(usage["ledger_attempt_ids"]) == 1

    @pytest.mark.parametrize("choice", [
        "none", "required", {"type": "function", "function": {"name": "probe"}},
    ])
    def test_explicit_tool_choice_survives_missing_catalog_support(self, monkeypatch, choice):
        from ouroboros.llm import LLMClient
        from ouroboros.request_wire_recovery import prepare_wire_payload_for_send, request_wire_call_scope

        client = LLMClient(api_key="test")
        monkeypatch.setattr(client, "_get_supported_parameters", lambda _model: {
            "tools", "reasoning", "max_tokens",
        })
        target = client._resolve_remote_target("anthropic/claude-fable-5.1")
        with request_wire_call_scope():
            kwargs = client._build_remote_kwargs(
                target=target, messages=[{"role": "user", "content": "probe"}],
                reasoning_effort="medium", max_tokens=65536, tool_choice=choice,
                temperature=None, tools=[{"type": "function", "function": {
                    "name": "probe", "parameters": {"type": "object", "properties": {}},
                }}],
            )
            physical = prepare_wire_payload_for_send(target, kwargs, api_surface="chat.completions")
        assert physical["tool_choice"] == choice
        assert physical["extra_body"]["reasoning"]["effort"] == "medium"

    def test_temperature_stripped_for_unsupported_model(self, monkeypatch):
        from ouroboros.llm import LLMClient
        from ouroboros.request_wire_recovery import (
            prepare_wire_payload_for_send,
            request_wire_call_scope,
        )

        _install_fake_response(monkeypatch, {
            "data": [{
                "id": "anthropic/claude-opus-4.6",
                "supported_parameters": [
                    "include_reasoning", "max_tokens", "reasoning",
                    "response_format", "stop", "structured_outputs",
                    "tool_choice", "tools", "verbosity",
                ],
            }]
        })

        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("anthropic/claude-opus-4.6")
        with request_wire_call_scope():
            kwargs = client._build_remote_kwargs(
                target=target,
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="medium",
                max_tokens=256,
                tool_choice="auto",
                temperature=0.2,
                tools=None,
            )
            physical = prepare_wire_payload_for_send(
                target, kwargs, api_surface="chat.completions"
            )
        assert kwargs["temperature"] == 0.2
        assert "temperature" not in physical

    def test_temperature_kept_for_supported_model(self, monkeypatch):
        from ouroboros.llm import LLMClient

        _install_fake_response(monkeypatch, {
            "data": [{
                "id": "anthropic/claude-opus-4.6",
                "supported_parameters": [
                    "include_reasoning", "max_tokens", "reasoning",
                    "response_format", "stop", "structured_outputs",
                    "temperature", "tool_choice", "tools", "top_k",
                    "top_p", "verbosity",
                ],
            }]
        })

        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("anthropic/claude-opus-4.6")
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=0.2,
            tools=None,
        )
        assert kwargs.get("temperature") == 0.2

    def test_fetch_failure_falls_back_to_no_stripping(self, monkeypatch):
        """When fetch fails (network/parse/missing), cache is empty and no params are stripped."""
        from ouroboros.llm import LLMClient

        def exploding_get(url: str, timeout: int = 15):
            raise RuntimeError("simulated transport failure")

        import requests
        monkeypatch.setattr(requests, "get", exploding_get)

        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("anthropic/claude-opus-4.6")
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=0.2,
            tools=None,
        )
        # The cache is empty → _get_supported_parameters returns None → no stripping.
        # Temperature survives (zero-regression fallback when offline).
        assert kwargs.get("temperature") == 0.2

    def test_main_openrouter_web_search_tool_is_opt_in(self, monkeypatch):
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH_ENGINE", "auto")
        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS", "10")
        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("openai/gpt-5.5")
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=[{"type": "function", "function": {"name": "noop_tool", "description": "noop", "parameters": {"type": "object", "properties": {}}}}],
            allow_server_web_search=True,
            skip_capability_fetch=True,
        )
        assert kwargs["tools"][-1] == {
            "type": "openrouter:web_search",
            "parameters": {"max_total_results": 10},
        }

    def test_main_openrouter_web_search_preserves_non_auto_engine(self, monkeypatch):
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH_ENGINE", "exa")
        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS", "4")
        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("openai/gpt-5.5")
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=[{"type": "function", "function": {"name": "noop_tool", "description": "noop", "parameters": {"type": "object", "properties": {}}}}],
            allow_server_web_search=True,
            skip_capability_fetch=True,
        )
        assert kwargs["tools"][-1] == {
            "type": "openrouter:web_search",
            "parameters": {"engine": "exa", "max_total_results": 4},
        }
        assert "search_context_size" not in kwargs["tools"][-1]

    def test_main_openrouter_web_search_respects_allow_flag(self, monkeypatch):
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("openai/gpt-5.5")
        tool = {"type": "function", "function": {"name": "noop_tool", "description": "noop", "parameters": {"type": "object", "properties": {}}}}
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=[tool],
            allow_server_web_search=False,
            skip_capability_fetch=True,
        )
        assert kwargs["tools"] == [tool]

    def test_chat_path_forwards_main_openrouter_web_search_flag(self, monkeypatch):
        from types import SimpleNamespace
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH_MAX_TOTAL_RESULTS", "2")
        captured = {}

        class _Completions:
            def create(self, **_kwargs):
                return None

        class _Client:
            chat = SimpleNamespace(completions=_Completions())

        class _Resp:
            def model_dump(self):
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}

        client = LLMClient(api_key="test")
        monkeypatch.setattr(client, "_get_remote_client", lambda _target: _Client())

        def fake_create(create_fn, kwargs, target):
            captured.update(kwargs)
            return _Resp()

        monkeypatch.setattr(client, "_create_chat_completion_with_retries", fake_create)
        tool = {"type": "function", "function": {"name": "noop_tool", "description": "noop", "parameters": {"type": "object", "properties": {}}}}
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.5",
            tools=[tool],
            allow_server_web_search=True,
        )
        assert captured["tools"][-1]["type"] == "openrouter:web_search"
        assert captured["tools"][-1]["parameters"]["max_total_results"] == 2

    def test_chat_no_proxy_path_forwards_main_openrouter_web_search_flag(self, monkeypatch):
        from types import SimpleNamespace
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
        captured = {}

        class _Completions:
            def create(self, **_kwargs):
                return None

        class _HttpClient:
            def close(self):
                pass

        class _Resp:
            def model_dump(self):
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}

        client = LLMClient(api_key="test")
        oa_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        monkeypatch.setattr(client, "_make_no_proxy_client", lambda _target, timeout=None: (oa_client, _HttpClient()))
        monkeypatch.setattr(client, "_create_chat_completion_with_retries", lambda _create_fn, kwargs, _target: (captured.update(kwargs) or _Resp()))
        tool = {"type": "function", "function": {"name": "noop_tool", "description": "noop", "parameters": {"type": "object", "properties": {}}}}
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.5",
            tools=[tool],
            no_proxy=True,
            allow_server_web_search=True,
        )
        assert captured["tools"][-1]["type"] == "openrouter:web_search"

    def test_main_openrouter_web_search_does_not_attach_to_toolless_calls(self, monkeypatch):
        from ouroboros.llm import LLMClient

        monkeypatch.setenv("OUROBOROS_MAIN_WEB_SEARCH", "openrouter")
        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("openai/gpt-5.5")
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "review"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=None,
            skip_capability_fetch=True,
        )
        assert "tools" not in kwargs

    def test_openrouter_web_annotations_surface_in_usage(self, monkeypatch):
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("openai/gpt-5.5")
        message, usage = client._normalize_remote_response({
            "choices": [{"message": {
                "role": "assistant",
                "content": "answer",
                "annotations": [{
                    "type": "url_citation",
                    "url_citation": {"url": "https://example.com", "title": "Example", "content": "snippet"},
                }],
            }}],
            "usage": {"server_tool_use": {"web_search_requests": 1}},
        }, target, skip_cost_fetch=True)
        assert message["content"] == "answer"
        assert "annotations" not in message
        assert usage["server_tool_use"]["web_search_requests"] == 1
        assert usage["web_search_sources"][0]["url"] == "https://example.com"

    def test_openrouter_web_search_server_tool_uses_parameters_shape(self, monkeypatch):
        from ouroboros.llm import openrouter_web_search_server_tool

        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return object()

        class _OpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = types.SimpleNamespace(completions=_Completions())

        monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))

        openrouter_web_search_server_tool(
            api_key="key",
            model="openai/gpt-5.5",
            query="q",
            search_context_size="medium",
        )

        assert captured["tools"] == [{
            "type": "openrouter:web_search",
            "parameters": {"search_context_size": "medium", "max_total_results": 10},
        }]
        from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS

        assert captured["client"]["default_headers"] == OPENROUTER_APP_HEADERS

    def test_parameter_rejection_learns_sampling_strip_without_version_gate(self, monkeypatch):
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("anthropic/claude-opus-4.8")
        first = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=0.2,
            tools=None,
            skip_capability_fetch=True,
        )
        assert first["model"] == "anthropic/claude-opus-4.8"
        assert first.get("temperature") == 0.2

        retry = client._retry_without_optional_sampling(
            first,
            "anthropic/claude-opus-4.8",
            RuntimeError("404 No endpoints found that can handle the requested parameters"),
        )
        assert retry is not None
        assert "temperature" not in retry
        assert retry["extra_body"]["reasoning"]["effort"] == "medium"
        assert retry["extra_body"]["provider"]["require_parameters"] is True

        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=0.2,
            tools=None,
            skip_capability_fetch=True,
        )

        # Legacy model-global rows stay diagnostic and cannot mutate dispatch.
        assert kwargs["temperature"] == 0.2

    def test_cache_fetched_at_most_once(self, monkeypatch):
        from ouroboros.llm import LLMClient

        call_count = _install_fake_response(monkeypatch, {
            "data": [{
                "id": "anthropic/claude-opus-4.6",
                "supported_parameters": ["max_tokens"],
            }]
        })

        client = LLMClient(api_key="test")
        target = client._resolve_remote_target("anthropic/claude-opus-4.6")
        # Two back-to-back calls: the second must hit the cache, not the network.
        for _ in range(2):
            client._build_remote_kwargs(
                target=target,
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="medium",
                max_tokens=256,
                tool_choice="auto",
                temperature=0.2,
                tools=None,
            )
        assert call_count["n"] == 1, (
            f"Expected supported_parameters fetch to run exactly once per process, "
            f"got {call_count['n']} calls"
        )

    def test_unknown_model_falls_back_to_no_stripping(self, monkeypatch):
        """A model missing from OpenRouter's /models list keeps all kwargs."""
        from ouroboros.llm import LLMClient

        _install_fake_response(monkeypatch, {
            "data": [{
                "id": "anthropic/claude-opus-4.6",
                "supported_parameters": ["max_tokens"],  # no temperature
            }]
        })

        client = LLMClient(api_key="test")
        # Query a model NOT in our fake response
        target = client._resolve_remote_target("anthropic/future-unknown-model")
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=0.2,
            tools=None,
        )
        # Unknown model → cache miss → None → no stripping
        assert kwargs.get("temperature") == 0.2

    def test_no_version_specific_sampling_gate_remains(self):
        import pathlib

        llm_text = pathlib.Path(__file__).resolve().parents[1].joinpath("ouroboros", "llm.py").read_text(encoding="utf-8")
        assert "claude-opus-4-7" not in llm_text
        assert "claude-opus-4.7" not in llm_text
