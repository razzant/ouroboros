"""Focused v6.64 prompt-cache affinity and fallback regressions."""

from __future__ import annotations

import asyncio
import copy
import json

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
    # Measured 2026-09-25 on openai/gpt-6-sol: OpenAI's public API reuses a prompt cache
    # only under ONE routing key, so the OpenAI family shares a session per model and
    # governance prefix — a different first user message keeps the SAME session_id
    # (llm_routing._openrouter_session_identity, the openai-family branch).
    assert kwargs["extra_body"]["session_id"] == other["extra_body"]["session_id"]
    assert kwargs["extra_body"]["reasoning"]["effort"] == "high"
    assert "prompt_cache_key" not in kwargs

    # Every other family keeps the conversation-stable session: the first user
    # message is folded in, so a different owner prompt is a different session.
    grok = {
        "provider": "openrouter",
        "resolved_model": "x-ai/grok-4.7",
        "usage_model": "x-ai/grok-4.7",
        "supports_openrouter_extensions": True,
    }

    def build_grok(messages):
        return client._build_remote_kwargs(
            grok, messages, "high", 512, "auto", None, None,
            skip_capability_fetch=True,
        )

    grok_kwargs = build_grok(_messages())
    assert grok_kwargs["extra_body"]["session_id"].startswith("ouroboros-session-")
    assert grok_kwargs["extra_body"]["session_id"] == build_grok(continued_messages)["extra_body"]["session_id"]
    assert grok_kwargs["extra_body"]["session_id"] != build_grok(different_owner_prompt)["extra_body"]["session_id"]
    assert grok_kwargs["extra_body"]["session_id"] != kwargs["extra_body"]["session_id"]
    assert "prompt_cache_key" not in grok_kwargs


def _sealed_transcripts():
    from ouroboros.context_fit import seal_task_transcript

    messages = _messages(stable="stable policy " * 800)
    snapshots = []
    for count in range(1, 8):
        messages.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"read-{count}", "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": f"read-{count}", "content": f"evidence {count}"},
        ])
        if count >= 5:
            seal_task_transcript(messages)
            snapshots.append(copy.deepcopy(messages))
    # A compacted transcript can return the cache boundary to the task message.
    messages = messages[:len(snapshots[0])]
    seal_task_transcript(messages)
    snapshots.append(messages)
    return snapshots


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("model,cache_markers", [
    ("openai/gpt-5.5", False), ("anthropic/claude-fable-5", True),
])
def test_openrouter_session_survives_marker_migration_on_sdk_wire(asynchronous, model, cache_markers):
    import httpx
    import openai
    from ouroboros.llm import LLMClient

    snapshots = _sealed_transcripts()
    originals = copy.deepcopy(snapshots)
    client = LLMClient(api_key="unused")
    target = {
        "provider": "openrouter", "resolved_model": model, "usage_model": model,
        "supports_openrouter_extensions": True,
    }
    tools = [{"type": "function", "function": {
        "name": "read_file", "parameters": {"type": "object", "properties": {}},
    }}]
    payloads = [client._build_remote_kwargs(
        target, messages, "high", 512, "auto", None, tools,
        skip_capability_fetch=True,
    ) for messages in snapshots]
    assert snapshots == originals, "affinity projection must not mutate the canonical transcript"
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                         "finish_reason": "stop"}],
        })

    async def send_async():
        async with openai.AsyncOpenAI(
            api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as sdk:
            for payload in payloads:
                await sdk.chat.completions.create(**payload)

    if asynchronous:
        asyncio.run(send_async())
    else:
        with openai.OpenAI(
            api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as sdk:
            for payload in payloads:
                sdk.chat.completions.create(**payload)

    assert len(captured) == 4
    assert len({payload["session_id"] for payload in captured}) == 1
    assert all(payload["tools"] == captured[0]["tools"] for payload in captured)
    boundaries = []
    for payload in captured:
        marked = [i for i, message in enumerate(payload["messages"])
                  if message["role"] != "system" and isinstance(message["content"], list)
                  for block in message["content"] if "cache_control" in block]
        boundaries.append(marked)
    if cache_markers:
        assert boundaries == [[1], [3], [5], [1]], "supported markers must still migrate both ways"
    else:
        assert boundaries == [[], [], [], []]
        for previous, current in zip(captured[:2], captured[1:3]):
            assert current["messages"][:len(previous["messages"])] == previous["messages"]
        assert captured[3]["messages"] == captured[0]["messages"]


def _first_user_variants(messages):
    """First-user rewrites that a conversation-stable session must tell apart:
    text (also a trailing-space change), the image url, the block order."""
    changed = []
    for text in ("another task", "solve exactly "):
        candidate = copy.deepcopy(messages)
        candidate[1]["content"][0]["text"] = text
        changed.append(candidate)
    candidate = copy.deepcopy(messages)
    candidate[1]["content"][1]["image_url"]["url"] = "data:image/png;base64,AQ=="
    changed.append(candidate)
    candidate = copy.deepcopy(messages)
    candidate[1]["content"].reverse()
    changed.append(candidate)
    return changed


def _transport_annotated(messages, ttl):
    annotated = copy.deepcopy(messages)
    annotated[1]["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": ttl}
    annotated[1]["content"][1].update({
        "_caption": "host caption", "_source_path": "/fixture/image.png", "_context_capsule": "host",
    })
    return annotated


def test_derived_session_ignores_transport_metadata_but_preserves_semantic_inputs():
    # Pinned on a non-OpenAI family since 2026-09-25: the OpenAI family's session is
    # prefix-only (see the inverse test below); every other family keeps folding the
    # first user message in, so the conversation-stable guarantees stay exactly these.
    from ouroboros.llm import LLMClient

    messages = _messages()
    messages[1]["content"] = [
        {"type": "text", "text": "solve exactly"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
    ]
    baseline = copy.deepcopy(messages)
    identity = LLMClient._openrouter_session_identity("anthropic/claude-fable-5", messages)
    assert identity.startswith("ouroboros-session-")
    for ttl in ("5m", "1h"):
        annotated = _transport_annotated(messages, ttl)
        original = copy.deepcopy(annotated)
        assert LLMClient._openrouter_session_identity("anthropic/claude-fable-5", annotated) == identity
        assert annotated == original
    changed = _first_user_variants(messages)
    candidate = copy.deepcopy(messages)
    candidate[0]["content"][0]["text"] = "another policy"
    changed.append(candidate)
    assert all(LLMClient._openrouter_session_identity("anthropic/claude-fable-5", candidate) != identity
               for candidate in changed)
    assert LLMClient._openrouter_session_identity("anthropic/claude-opus-5", messages) != identity
    assert messages == baseline


def test_openai_family_session_ignores_the_first_user_message_but_tracks_prefix_and_model():
    """The inverse of the test above for OpenAI's family (measured 2026-09-25 on
    openai/gpt-6-sol: one routing key, whole-section cache unit). The first user
    message — its text, image url, block order, host metadata and cache markers — never
    enters the session, while block 0 of the system prefix and the model still do.
    Removing the openai-family branch of ``_openrouter_session_identity`` fails the
    equality half; hashing only the model would fail the block-0 half."""
    from ouroboros.llm import LLMClient

    messages = _messages()
    messages[1]["content"] = [
        {"type": "text", "text": "solve exactly"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"}},
    ]
    baseline = copy.deepcopy(messages)
    identity = LLMClient._openrouter_session_identity("openai/gpt-5.5", messages)
    assert identity.startswith("ouroboros-session-")
    same_session = _first_user_variants(messages) + [_transport_annotated(messages, "5m"), _transport_annotated(messages, "1h")]
    assert all(LLMClient._openrouter_session_identity("openai/gpt-5.5", candidate) == identity
               for candidate in same_session)
    candidate = copy.deepcopy(messages)
    candidate[0]["content"][0]["text"] = "another policy"
    assert LLMClient._openrouter_session_identity("openai/gpt-5.5", candidate) != identity
    assert LLMClient._openrouter_session_identity("openai/gpt-5.6-sol", messages) != identity
    assert LLMClient._openrouter_session_identity("anthropic/claude-fable-5", messages) != identity
    assert messages == baseline


def test_named_openai_cache_parameter_gets_one_exact_retry(monkeypatch):
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
    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient

    async def passthrough(_request, send):
        return await send()

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt_async", passthrough)
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


def test_generic_403_does_not_trigger_cache_retry_or_provider_hop(monkeypatch):
    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(
        llm_attempt_mod,
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
