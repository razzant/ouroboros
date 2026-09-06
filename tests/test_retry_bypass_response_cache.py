"""Regression tests: a retried LLM call must not be served from a response cache.

`provider_incomplete_response` is classified as transient (`_TRANSIENT_RETRY_KINDS`),
i.e. the retry loop assumes a repeat MAY produce a different result.  That assumption
only holds when nothing between the client and the model caches responses.  A gateway
response cache (LiteLLM `cache: true`) replays the identical failed body for every
attempt, so the whole transient-retry budget is spent without ever reaching the model
and the task ends as `infra_failed` / `provider_unavailable`.

Observed in the field: six attempts, one shared `response_id`, each returned in ~0.0s;
the same request replayed later succeeded.
"""

from __future__ import annotations

import pytest


class TestBuildRemoteKwargsCacheOptOut:
    def test_no_cache_field_absent_by_default(self):
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test")
        target = {
            "provider": "openai-compatible",
            "resolved_model": "local-reason",
            "usage_model": "local-reason",
            "api_key": "test",
            "base_url": "http://127.0.0.1:4000/v1",
            "default_headers": {},
        }
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=None,
        )
        assert "cache" not in (kwargs.get("extra_body") or {}), (
            "the first attempt must stay cacheable; only retries opt out"
        )
        assert "cache" not in kwargs, "cache must never be a top-level kwarg"

    def test_cache_field_present_when_bypassing(self):
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test")
        target = {
            "provider": "openai-compatible",
            "resolved_model": "local-reason",
            "usage_model": "local-reason",
            "api_key": "test",
            "base_url": "http://127.0.0.1:4000/v1",
            "default_headers": {},
        }
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=None,
            bypass_response_cache=True,
        )
        assert (kwargs.get("extra_body") or {}).get("cache") == {"no-cache": True}, (
            "a retry must carry LiteLLM's documented per-request cache opt-out, "
            "otherwise the gateway replays the cached failed response"
        )
        assert "cache" not in kwargs, (
            "the OpenAI SDK raises TypeError on unknown top-level kwargs, so the "
            "opt-out must ride in extra_body"
        )

    def test_litellm_control_is_not_sent_to_direct_providers_or_openrouter(self):
        from ouroboros.llm import LLMClient

        client = LLMClient(api_key="test")
        targets = [
            {
                "provider": "openai",
                "resolved_model": "gpt-5.5",
                "usage_model": "openai/gpt-5.5",
                "supports_openrouter_extensions": False,
            },
            {
                "provider": "minimax",
                "resolved_model": "MiniMax-M2.5",
                "usage_model": "minimax/MiniMax-M2.5",
                "supports_openrouter_extensions": False,
            },
            {
                "provider": "cloudru",
                "resolved_model": "foundation-model",
                "usage_model": "cloudru/foundation-model",
                "supports_openrouter_extensions": False,
            },
            {
                "provider": "openrouter",
                "resolved_model": "openai/gpt-5.5",
                "usage_model": "openai/gpt-5.5",
                "supports_openrouter_extensions": True,
            },
        ]

        for target in targets:
            kwargs = client._build_remote_kwargs(
                target=target,
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort="medium",
                max_tokens=256,
                tool_choice="auto",
                temperature=None,
                tools=None,
                skip_capability_fetch=True,
                bypass_response_cache=True,
            )
            assert "cache" not in (kwargs.get("extra_body") or {}), target["provider"]


class TestRetryLoopRequestsCacheOptOut:
    def test_incomplete_response_arms_only_the_following_attempt(self, tmp_path, monkeypatch):
        import time

        from ouroboros.loop_llm_call import call_llm_with_retry

        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        calls = []

        class IncompleteThenSuccessfulLLM:
            def chat(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return {"content": "", "tool_calls": [], "finish_reason": None}, {}
                return (
                    {"content": "recovered", "tool_calls": [], "finish_reason": "stop"},
                    {"provider": "openai-compatible", "resolved_model": "local-reason"},
                )

        msg, _cost = call_llm_with_retry(
            IncompleteThenSuccessfulLLM(),
            [{"role": "user", "content": "hi"}],
            "openai-compatible::local-reason",
            None,
            "medium",
            2,
            tmp_path,
            "task-incomplete-cache",
            1,
            None,
            {},
        )

        assert msg["content"] == "recovered"
        assert [call["bypass_response_cache"] for call in calls] == [False, True]

    def test_transport_retry_does_not_request_litellm_cache_bypass(self, tmp_path, monkeypatch):
        import time

        from ouroboros.loop_llm_call import call_llm_with_retry

        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        calls = []

        class TransientThenSuccessfulLLM:
            def chat(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    error = RuntimeError("503 service unavailable")
                    error.status_code = 503
                    raise error
                return (
                    {"content": "recovered", "tool_calls": [], "finish_reason": "stop"},
                    {"provider": "openai-compatible", "resolved_model": "local-reason"},
                )

        msg, _cost = call_llm_with_retry(
            TransientThenSuccessfulLLM(),
            [{"role": "user", "content": "hi"}],
            "openai-compatible::local-reason",
            None,
            "medium",
            2,
            tmp_path,
            "task-transient-cache",
            1,
            None,
            {},
        )

        assert msg["content"] == "recovered"
        assert [call["bypass_response_cache"] for call in calls] == [False, False]

    @pytest.mark.parametrize(("kind", "code"), [("rate_limit", 429), ("provider_transient", 503)])
    def test_transient_body_error_does_not_request_litellm_cache_bypass(
        self, kind, code, tmp_path, monkeypatch,
    ):
        import time

        from ouroboros.llm import LLMClient
        from ouroboros.loop_llm_call import call_llm_with_retry

        monkeypatch.setattr(time, "sleep", lambda _seconds: None)
        target = {
            "provider": "openai-compatible",
            "resolved_model": "local-reason",
            "usage_model": "openai-compatible/local-reason",
            "supports_openrouter_extensions": False,
        }
        body_error_msg, body_error_usage = LLMClient(api_key="test")._normalize_remote_response(
            {
                "id": f"body-error-{code}",
                "choices": [],
                "error": {"code": code, "message": "transient provider error"},
                "usage": {},
            },
            target,
            skip_cost_fetch=True,
        )
        assert body_error_usage["provider_error"]["kind"] == kind
        calls = []

        class BodyErrorThenSuccessfulLLM:
            def chat(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    return body_error_msg, body_error_usage
                return (
                    {"content": "recovered", "tool_calls": [], "finish_reason": "stop"},
                    {"provider": "openai-compatible", "resolved_model": "local-reason"},
                )

        msg, _cost = call_llm_with_retry(
            BodyErrorThenSuccessfulLLM(),
            [{"role": "user", "content": "hi"}],
            "openai-compatible::local-reason",
            None,
            "medium",
            2,
            tmp_path,
            f"task-body-error-{code}",
            1,
            None,
            {},
        )

        assert msg["content"] == "recovered"
        assert [call["bypass_response_cache"] for call in calls] == [False, False]


class TestStrictCompatibleRecovery:
    def test_explicit_cache_rejection_gets_one_exact_retry(self, monkeypatch):
        import ouroboros.llm_attempt as llm_attempt_mod
        from ouroboros.llm import LLMClient

        monkeypatch.setattr(
            llm_attempt_mod,
            "execute_physical_attempt",
            lambda _request, send: send(),
        )
        client = LLMClient(api_key="unused")
        calls = []
        expected = object()

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("400 unknown parameter: cache")
            return expected

        target = {
            "provider": "openai-compatible",
            "resolved_model": "strict-model",
            "usage_model": "openai-compatible/strict-model",
            "supports_openrouter_extensions": False,
        }
        kwargs = {
            "model": "strict-model",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {
                "cache": {"no-cache": True},
                "future_extension": {"keep": True},
            },
        }

        result = client._create_chat_completion_with_retries(create, kwargs, target)

        assert result is expected
        assert len(calls) == 2
        assert calls[0]["extra_body"]["cache"] == {"no-cache": True}
        assert "cache" not in calls[1]["extra_body"]
        assert calls[1]["extra_body"]["future_extension"] == {"keep": True}


class TestChatSignatureThreadsFlag:
    def test_chat_accepts_bypass_response_cache(self):
        import inspect

        from ouroboros.llm import LLMClient

        params = inspect.signature(LLMClient.chat).parameters
        assert "bypass_response_cache" in params, (
            "LLM.chat must expose the flag so the retry loop can request a fresh call"
        )
        assert params["bypass_response_cache"].default is False, (
            "bypassing the cache must be opt-in"
        )
