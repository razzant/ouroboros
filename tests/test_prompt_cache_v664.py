"""Focused v6.64 prompt-cache affinity and fallback regressions."""

from __future__ import annotations

import asyncio

import pytest


def _messages(stable: str = "stable policy", dynamic: str = "task one"):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": dynamic},
            ],
        },
        {"role": "user", "content": "solve"},
    ]


def test_direct_openai_cache_key_tracks_stable_prefix_not_dynamic_evidence(monkeypatch):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="unused")
    target = {
        "provider": "openai",
        "resolved_model": "gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": False,
    }

    def build(messages):
        return client._build_remote_kwargs(
            target, messages, "high", 512, "auto", None, None,
            skip_capability_fetch=True,
        )

    first = build(_messages(dynamic="task one"))
    same_prefix = build(_messages(dynamic="task two"))
    changed_prefix = build(_messages(stable="different policy", dynamic="task one"))

    assert first["prompt_cache_key"].startswith("ouroboros-")
    assert first["prompt_cache_key"] == same_prefix["prompt_cache_key"]
    assert first["prompt_cache_key"] != changed_prefix["prompt_cache_key"]
    assert "session_id" not in first.get("extra_body", {})


def test_openrouter_uses_session_id_without_replacing_existing_extra_body(monkeypatch):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="unused")
    target = {
        "provider": "openrouter",
        "resolved_model": "openai/gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": True,
    }
    kwargs = client._build_remote_kwargs(
        target, _messages(), "high", 512, "auto", None, None,
        skip_capability_fetch=True,
    )
    continued_messages = _messages() + [{"role": "assistant", "content": "working"}]
    continued = client._build_remote_kwargs(
        target, continued_messages, "high", 512, "auto", None, None,
        skip_capability_fetch=True,
    )
    different_owner_prompt = _messages()
    different_owner_prompt[1] = {"role": "user", "content": "another task"}
    other = client._build_remote_kwargs(
        target, different_owner_prompt, "high", 512, "auto", None, None,
        skip_capability_fetch=True,
    )

    assert kwargs["extra_body"]["session_id"].startswith("ouroboros-session-")
    assert kwargs["extra_body"]["session_id"] == continued["extra_body"]["session_id"]
    assert kwargs["extra_body"]["session_id"] != other["extra_body"]["session_id"]
    assert kwargs["extra_body"]["reasoning"]["effort"] == "high"
    assert "prompt_cache_key" not in kwargs


def test_named_openai_cache_parameter_gets_one_exact_retry(monkeypatch):
    import ouroboros.llm as llm_mod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(
        llm_mod,
        "execute_physical_attempt",
        lambda _request, send: send(),
    )
    client = LLMClient(api_key="unused")
    calls = []
    expected = object()

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TypeError(
                "Completions.create() got an unexpected keyword argument 'prompt_cache_key'"
            )
        return expected

    target = {
        "provider": "openai",
        "resolved_model": "gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": False,
    }
    kwargs = {
        "model": "gpt-5.5",
        "messages": _messages(),
        "max_completion_tokens": 10,
        "prompt_cache_key": "ouroboros-test",
    }

    assert client._create_chat_completion_with_retries(create, kwargs, target) is expected
    assert len(calls) == 2
    assert calls[0]["prompt_cache_key"] == "ouroboros-test"
    assert "prompt_cache_key" not in calls[1]
    assert calls[1]["messages"] == calls[0]["messages"]


def test_named_openrouter_session_id_gets_one_exact_async_retry(monkeypatch):
    import ouroboros.llm as llm_mod
    from ouroboros.llm import LLMClient

    async def passthrough(_request, send):
        return await send()

    monkeypatch.setattr(llm_mod, "execute_physical_attempt_async", passthrough)
    client = LLMClient(api_key="unused")
    calls = []
    expected = object()

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("400 unknown parameter: session_id")
        return expected

    target = {
        "provider": "openrouter",
        "resolved_model": "openai/gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": True,
    }
    kwargs = {
        "model": "openai/gpt-5.5",
        "messages": _messages(),
        "extra_body": {
            "session_id": "ouroboros-test",
            "reasoning": {"effort": "high"},
            "provider": {"allow_fallbacks": False},
        },
    }

    result = asyncio.run(
        client._create_chat_completion_with_retries_async(create, kwargs, target)
    )
    assert result is expected
    assert len(calls) == 2
    assert calls[0]["extra_body"]["session_id"] == "ouroboros-test"
    assert "session_id" not in calls[1]["extra_body"]
    assert calls[1]["extra_body"]["reasoning"] == {"effort": "high"}
    assert calls[1]["extra_body"]["provider"] == {"allow_fallbacks": False}


def test_unknown_outcome_does_not_retry_named_openai_cache_parameter(
    tmp_path, monkeypatch,
):
    from ouroboros.llm import LLMClient
    from ouroboros.usage_accounting import physical_provider_outcome_unknown

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "data" / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    client = LLMClient(api_key="unused")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("prompt_cache_key is not supported; connection dropped")

    target = {
        "provider": "openai",
        "resolved_model": "gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": False,
    }
    with pytest.raises(ConnectionError) as exc_info:
        client._create_chat_completion_with_retries(
            create,
            {
                "model": "gpt-5.5",
                "messages": _messages(),
                "prompt_cache_key": "ouroboros-test",
            },
            target,
        )

    assert physical_provider_outcome_unknown(exc_info.value) is True
    assert len(calls) == 1


def test_unknown_outcome_does_not_retry_named_openrouter_session_async(
    tmp_path, monkeypatch,
):
    from ouroboros.llm import LLMClient
    from ouroboros.usage_accounting import physical_provider_outcome_unknown

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "data" / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    client = LLMClient(api_key="unused")
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("session_id is unsupported; connection dropped")

    target = {
        "provider": "openrouter",
        "resolved_model": "openai/gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": True,
    }
    with pytest.raises(ConnectionError) as exc_info:
        asyncio.run(client._create_chat_completion_with_retries_async(
            create,
            {
                "model": "openai/gpt-5.5",
                "messages": _messages(),
                "extra_body": {"session_id": "ouroboros-test"},
            },
            target,
        ))

    assert physical_provider_outcome_unknown(exc_info.value) is True
    assert len(calls) == 1


def test_typed_http_cache_rejection_still_gets_safe_retry(tmp_path, monkeypatch):
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "data" / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    client = LLMClient(api_key="unused")
    calls = []
    expected = object()

    class Response:
        status_code = 400

        @staticmethod
        def json():
            return {"error": {"code": "invalid_parameter"}}

    class Rejection(Exception):
        def __init__(self):
            super().__init__("prompt_cache_key is an unsupported parameter")
            self.response = Response()

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Rejection()
        return expected

    target = {
        "provider": "openai",
        "resolved_model": "gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": False,
    }
    result = client._create_chat_completion_with_retries(
        create,
        {
            "model": "gpt-5.5",
            "messages": _messages(),
            "prompt_cache_key": "ouroboros-test",
        },
        target,
    )

    assert result is expected
    assert len(calls) == 2
    assert "prompt_cache_key" not in calls[1]


def test_generic_403_does_not_trigger_cache_retry_or_provider_hop(monkeypatch):
    import ouroboros.llm as llm_mod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(
        llm_mod,
        "execute_physical_attempt",
        lambda _request, send: send(),
    )
    client = LLMClient(api_key="unused")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("403 forbidden by account policy")

    target = {
        "provider": "openai",
        "resolved_model": "gpt-5.5",
        "usage_model": "openai/gpt-5.5",
        "supports_openrouter_extensions": False,
    }
    with pytest.raises(RuntimeError, match="403 forbidden"):
        client._create_chat_completion_with_retries(
            create,
            {
                "model": "gpt-5.5",
                "messages": _messages(),
                "prompt_cache_key": "ouroboros-test",
            },
            target,
        )

    assert len(calls) == 1


def test_existing_same_model_reroute_rotates_sticky_session():
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="unused")
    target = {
        "provider": "openrouter",
        "supports_openrouter_extensions": True,
    }
    kwargs = {
        "model": "openai/gpt-5.5",
        "messages": [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "working", "reasoning": "private"},
        ],
        "extra_body": {
            "session_id": "ouroboros-session-original",
            "reasoning": {"effort": "high"},
        },
    }

    rerouted = client._reroute_same_model_kwargs(
        target,
        kwargs,
        allow_portable_reasoning=True,
    )
    assert rerouted is not None
    assert rerouted["extra_body"]["session_id"].startswith("ouroboros-session-")
    assert rerouted["extra_body"]["session_id"] != kwargs["extra_body"]["session_id"]
    assert kwargs["extra_body"]["session_id"] == "ouroboros-session-original"
