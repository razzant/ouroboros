import pytest

from ouroboros.llm import LLMClient
from ouroboros.openrouter_attribution import OPENROUTER_APP_HEADERS


def test_resolve_openrouter_target_uses_canonical_app_attribution():
    target = LLMClient()._resolve_remote_target("openai/gpt-5.6-sol")

    assert OPENROUTER_APP_HEADERS == {
        "HTTP-Referer": "https://ouroboros-agent.ai/",
        "X-OpenRouter-Title": "Ouroboros",
    }
    assert target["default_headers"] == OPENROUTER_APP_HEADERS


def test_resolve_openai_target(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    target = LLMClient()._resolve_remote_target("openai::gpt-4.1")

    assert target["provider"] == "openai"
    assert target["resolved_model"] == "gpt-4.1"
    assert target["usage_model"] == "openai/gpt-4.1"
    assert target["base_url"] == "https://api.openai.com/v1"


def test_build_remote_kwargs_uses_max_completion_tokens_for_openai_gpt5(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs


def test_build_remote_kwargs_uses_current_carriers_provider_wide_for_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-4.1")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert kwargs["max_completion_tokens"] == 512
    assert kwargs["reasoning_effort"] == "high"
    assert "max_tokens" not in kwargs


def test_build_remote_kwargs_normalizes_tool_descriptions_for_openrouter():
    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")

    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        [{
            "type": "function",
            "function": {
                "name": "bad_tool",
                "description": ("first half ", "second half"),
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )

    assert kwargs["tools"][0]["function"]["description"] == "first half second half"


def test_build_remote_kwargs_deduplicates_tool_names_for_openrouter():
    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")

    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        [
            {
                "type": "function",
                "function": {
                    "name": "dup_tool",
                    "description": "first",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dup_tool",
                    "description": "second",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    )

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["dup_tool"]
    assert kwargs["tools"][0]["function"]["description"] == "first"


def test_openrouter_reasoning_returned_by_default_and_env_disables(monkeypatch):
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")

    monkeypatch.delenv("OUROBOROS_RETURN_REASONING", raising=False)
    default_kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert default_kwargs["extra_body"]["reasoning"] == {"effort": "high", "exclude": False}

    monkeypatch.setenv("OUROBOROS_RETURN_REASONING", "false")
    disabled_kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert disabled_kwargs["extra_body"]["reasoning"] == {"effort": "high", "exclude": True}

    monkeypatch.setenv("OUROBOROS_RETURN_REASONING", "")
    empty_disabled_kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert empty_disabled_kwargs["extra_body"]["reasoning"] == {"effort": "high", "exclude": True}


def test_non_openrouter_payload_strips_reasoning_roundtrip_metadata(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    client = LLMClient()
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            "reasoning": "read first",
            "reasoning_details": [{"type": "reasoning.text", "text": "read first"}],
            "response_id": "gen-123",
        },
    ]

    kwargs = client._build_remote_kwargs(
        client._resolve_remote_target("openai::gpt-5.2"),
        messages,
        "medium",
        512,
        "auto",
        None,
        None,
    )

    assistant_msg = kwargs["messages"][1]
    assert "reasoning" not in assistant_msg
    assert "reasoning_details" not in assistant_msg
    assert "response_id" not in assistant_msg
    assert messages[1]["reasoning_details"] == [{"type": "reasoning.text", "text": "read first"}]


def test_openrouter_payload_keeps_reasoning_roundtrip_metadata(monkeypatch):
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            "reasoning": "read first",
            "reasoning_details": [{"type": "reasoning.text", "text": "read first"}],
            "response_id": "gen-123",
        },
    ]

    kwargs = client._build_remote_kwargs(
        client._resolve_remote_target("anthropic/claude-sonnet-4.6"),
        messages,
        "medium",
        512,
        "auto",
        None,
        None,
    )

    assistant_msg = kwargs["messages"][1]
    assert assistant_msg["reasoning"] == "read first"
    assert assistant_msg["reasoning_details"] == [{"type": "reasoning.text", "text": "read first"}]
    assert assistant_msg["response_id"] == "gen-123"


def test_system_message_placement_demotes_late_notices_preserving_tool_adjacency():
    client = LLMClient()
    messages = [
        {"role": "system", "content": "authoritative"},
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "system", "content": "late notice"},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "user", "content": "continue"},
    ]

    normalized = client._normalize_system_message_placement(messages)

    assert [m["role"] for m in normalized] == ["system", "user", "assistant", "tool", "user", "user"]
    assert normalized[0]["content"] == "authoritative"
    assert normalized[3]["role"] == "tool"
    assert normalized[4]["content"].startswith("[SYSTEM NOTICE]\nlate notice")


def test_build_remote_kwargs_never_sends_non_leading_system_to_strict_providers(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compatible.example/v1")
    client = LLMClient()
    target = client._resolve_remote_target("openai-compatible::qwen-test")

    kwargs = client._build_remote_kwargs(
        target,
        [
            {"role": "system", "content": "root"},
            {"role": "user", "content": "start"},
            {"role": "system", "content": "late"},
        ],
        "medium",
        512,
        "auto",
        None,
        None,
    )

    assert [m["role"] for m in kwargs["messages"]] == ["system", "user", "user"]
    assert kwargs["messages"][2]["content"].startswith("[SYSTEM NOTICE]\nlate")


def test_openrouter_reasoning_details_disable_provider_fallbacks(monkeypatch):
    """A SEALED artifact (here an encrypted block on ``z-ai/glm`` — the original
    GLM->Claude cross-family signature bug) keeps the conservative
    ``allow_fallbacks=false`` pin, so an endpoint-bound artifact cannot silently fail
    over to a sibling provider. Readable artifacts stay failover-eligible — see
    ``test_portable_reasoning_replay_stays_failover_eligible``."""
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "thinking",
            "reasoning_details": [{"type": "reasoning.encrypted", "data": "sig"}],
        },
    ]

    kwargs = client._build_remote_kwargs(
        client._resolve_remote_target("z-ai/glm-4.6"),
        messages,
        "medium",
        512,
        "auto",
        None,
        None,
    )

    assert kwargs["extra_body"]["provider"]["allow_fallbacks"] is False
    assert kwargs["messages"][1]["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "sig"}]


def test_unverified_family_signed_reasoning_block_keeps_pin(monkeypatch):
    """The pin sees CONTENT blocks, not just top-level ``reasoning_details``: a
    non-roster family carrying a SIGNED thinking block with NO ``reasoning_details``
    must STILL be pinned, else an endpoint-bound signature could fail over to a sibling
    provider through a non-``reasoning_details`` artifact."""
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ],
        },
    ]

    kwargs = client._build_remote_kwargs(
        client._resolve_remote_target("z-ai/glm-4.6"),
        messages,
        "medium",
        512,
        "auto",
        None,
        None,
    )

    assert kwargs["extra_body"]["provider"]["allow_fallbacks"] is False


_SIGNED_TEXT = [{"type": "reasoning.text", "text": "t", "signature": "sig"}]
_PLAIN_TEXT = [{"type": "reasoning.text", "text": "t"}]


def _replay_assistant_turn(**extra):
    return [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "thinking", **extra},
    ]


def _dispatch_provider_block(monkeypatch, model, messages):
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    kwargs = client._build_remote_kwargs(
        client._resolve_remote_target(model),
        messages,
        "medium",
        512,
        "auto",
        None,
        None,
    )
    return kwargs, kwargs["extra_body"].get("provider", {})


@pytest.mark.parametrize("model,details", [
    # Roster-vouched families: their SIGNED artifacts replay across sibling endpoints.
    ("anthropic/claude-sonnet-4.6", _SIGNED_TEXT),
    ("google/gemini-3.5-flash", _SIGNED_TEXT),
    # Shape-first: an UNSIGNED readable artifact is portable for EVERY family — the
    # issue #468 case (a deepseek run pinned to one 429ing endpoint with 30 healthy ones).
    ("deepseek/deepseek-v4-flash-0731", _PLAIN_TEXT),
    ("z-ai/glm-4.6", _PLAIN_TEXT),
    ("openai/gpt-5.5", _PLAIN_TEXT),
])
def test_portable_reasoning_replay_stays_failover_eligible(monkeypatch, model, details):
    """A replayed artifact that is NOT sealed must never pin ``allow_fallbacks=false``:
    same-model provider failover keeps continuity and stays eligible under an upstream
    rate-limit. The anthropic cache route still pins ``require_parameters``; the others
    add no provider block at all. The replayed reasoning is preserved either way."""
    kwargs, provider = _dispatch_provider_block(
        monkeypatch, model, _replay_assistant_turn(reasoning_details=details),
    )

    assert "allow_fallbacks" not in provider  # same-model provider failover stays eligible
    if model.startswith("anthropic/"):
        assert provider.get("require_parameters") is True
    assert kwargs["messages"][1]["reasoning_details"] == details


def test_bare_response_id_no_longer_pins_the_provider(monkeypatch):
    """``response_id`` is stamped on EVERY OpenRouter response, so treating it as a
    replay artifact pinned every round after the first — for a transcript carrying no
    reasoning at all. It is an identifier, not an artifact: no pin (issue #468)."""
    _kwargs, provider = _dispatch_provider_block(
        monkeypatch, "deepseek/deepseek-v4-flash-0731",
        _replay_assistant_turn(response_id="gen-1"),
    )

    assert "allow_fallbacks" not in provider


def test_openai_encrypted_reasoning_pins_the_provider(monkeypatch):
    """Field 2026-07 (gpt-5.6-sol over 3x OpenAI + 2x Azure): replayed ENCRYPTED items
    answered "The encrypted content for item rs_... could not be ..." with a 400 after
    429-driven reroutes and killed benchmark waves. The reactive reroute path already
    stripped for openai/*; the proactive dispatch pin used to let it through. Both now
    read the same artifact-shape truth, so openai/* encrypted reasoning pins."""
    _kwargs, provider = _dispatch_provider_block(
        monkeypatch, "openai/gpt-5.5",
        _replay_assistant_turn(
            reasoning_details=[{"type": "reasoning.encrypted", "data": "rs_blob"}],
        ),
    )

    assert provider["allow_fallbacks"] is False


@pytest.mark.parametrize("model,pinned", [
    ("openai/gpt-5.5", True),
    ("google/gemini-3.5-flash", False),
])
def test_mixed_artifacts_seal_on_the_worst_form(monkeypatch, model, pinned):
    """A transcript is sealed if ANY carried artifact is — a readable summary sitting
    next to an encrypted blob does not make the turn replayable. The roster still
    vouches every form for its own families."""
    _kwargs, provider = _dispatch_provider_block(
        monkeypatch, model,
        _replay_assistant_turn(reasoning_details=[
            {"type": "reasoning.summary", "summary": "s"},
            {"type": "reasoning.encrypted", "data": "blob"},
        ]),
    )

    assert (provider.get("allow_fallbacks") is False) is pinned


def test_body_error_reroute_preserves_reasoning_for_portable_family():
    """On a TRANSIENT 200-body provider error (the rate-limit reroute path the failover
    exists for), a verified-portable family KEEPS its replayed reasoning across the
    same-model sibling-provider switch (the signature is portable), while an unverified
    family still strips. The 400 signature-REJECTION path strips for every family."""
    def _mk(model):
        return {
            "model": model,
            "messages": [
                {"role": "assistant", "reasoning_details": [{"type": "reasoning.text", "signature": "s"}]},
                {"role": "user", "content": "hi"},
            ],
            "extra_body": {"provider": {}},
        }

    inst = LLMClient.__new__(LLMClient)
    target = {"supports_openrouter_extensions": True}

    # transient body-error path (allow_portable_reasoning=True): portable family preserved
    portable = inst._reroute_same_model_kwargs(
        target, _mk("google/gemini-3.5-flash"), allow_portable_reasoning=True
    )
    assert portable is not None
    assert LLMClient._has_replayed_reasoning_metadata(portable["messages"]) is True

    # transient body-error path: unverified family still strips
    unverified = inst._reroute_same_model_kwargs(
        target, _mk("z-ai/glm-4.6"), allow_portable_reasoning=True
    )
    assert LLMClient._has_replayed_reasoning_metadata(unverified["messages"]) is False

    # 400 signature-rejection path (default allow_portable_reasoning=False): strips even portable
    sig400 = inst._reroute_same_model_kwargs(target, _mk("google/gemini-3.5-flash"))
    assert LLMClient._has_replayed_reasoning_metadata(sig400["messages"]) is False


def test_body_error_reroute_preserves_plain_reasoning_for_any_family():
    """The rate-limit reroute keeps an UNSIGNED readable artifact for a non-roster
    family too — its portability is a property of the artifact, not of the vendor
    (issue #468). The sticky session key is still rotated: replaying the old
    ``session_id`` would land the retry back on the endpoint that just 429'd."""
    inst = LLMClient.__new__(LLMClient)
    kwargs = {
        "model": "deepseek/deepseek-v4-flash-0731",
        "messages": [
            {"role": "assistant", "reasoning_details": [{"type": "reasoning.text", "text": "t"}]},
            {"role": "user", "content": "hi"},
        ],
        "extra_body": {"provider": {}, "session_id": "ouroboros-session-old"},
    }

    out = inst._reroute_same_model_kwargs(
        {"supports_openrouter_extensions": True}, kwargs, allow_portable_reasoning=True,
    )

    assert out is not None
    assert out["messages"][0]["reasoning_details"] == [{"type": "reasoning.text", "text": "t"}]
    assert out["extra_body"]["session_id"] != "ouroboros-session-old"
    assert out["extra_body"]["session_id"].startswith("ouroboros-session-")


def test_openrouter_signature_error_retries_once_with_reasoning_stripped(monkeypatch):
    client = LLMClient()
    target = client._resolve_remote_target("google/gemini-3.5-flash")
    kwargs = {
        "model": "google/gemini-3.5-flash",
        "extra_body": {"provider": {"allow_fallbacks": False}},
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "thinking",
                "reasoning": "private",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "sig"}],
                "response_id": "gen-1",
            },
        ],
    }
    calls = []

    class _Resp:
        def model_dump(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def fake_create(**call_kwargs):
        calls.append(call_kwargs)
        if len(calls) == 1:
            raise RuntimeError("400 INVALID_ARGUMENT: Corrupted thought signature")
        return _Resp()

    resp = client._create_chat_completion_with_retries(fake_create, kwargs, target)

    assert resp.model_dump()["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    retried_assistant = calls[1]["messages"][1]
    assert "reasoning" not in retried_assistant
    assert "reasoning_details" not in retried_assistant
    assert "response_id" not in retried_assistant
    assert "extra_body" not in calls[1]


def test_openrouter_encrypted_reasoning_item_error_triggers_same_strip_retry():
    """gpt-5-style 400s about encrypted reasoning items replayed from the
    transcript must reuse the existing strip-and-retry (same model, once)."""
    client = LLMClient()
    target = client._resolve_remote_target("openai/gpt-5.5")
    kwargs = {
        "model": "openai/gpt-5.5",
        "extra_body": {"provider": {"allow_fallbacks": False}},
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "thinking",
                "reasoning": "private",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "rs_abc"}],
                "response_id": "gen-2",
            },
        ],
    }
    calls = []

    class _Resp:
        def model_dump(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def fake_create(**call_kwargs):
        calls.append(call_kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "Error code: 400 - Could not load the encrypted content for item rs_abc."
            )
        return _Resp()

    resp = client._create_chat_completion_with_retries(fake_create, kwargs, target)

    assert resp.model_dump()["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    retried_assistant = calls[1]["messages"][1]
    assert "reasoning" not in retried_assistant
    assert "reasoning_details" not in retried_assistant
    assert "response_id" not in retried_assistant


def test_openrouter_gemini_preserves_message_cache_blocks_and_strips_tool_cache(monkeypatch):
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "dynamic"},
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": "alpha_tool",
            "description": "a",
            "parameters": {"type": "object", "properties": {}},
        },
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }]

    kwargs = client._build_remote_kwargs(
        client._resolve_remote_target("google/gemini-3.5-flash"),
        messages,
        "medium",
        512,
        "auto",
        None,
        tools,
    )

    assert isinstance(kwargs["messages"][0]["content"], list)
    # v6.69.0: ttl passthrough is ROUTE-GATED — anthropic/* keeps a valid
    # "5m"/"1h" ttl, while the Gemini route collapses to the bare marker
    # (its explicit cache documents no ttl field; an unknown field risks a
    # hard 400). Anthropic-route behavior is pinned in
    # tests/test_review_prompt_caching.py.
    assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in kwargs["tools"][0]
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert tools[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    messages = [
        {"role": "tool", "tool_call_id": "call-1", "content": [
            {"type": "text", "text": "cached result", "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
        ]},
    ]

    or_target = client._resolve_remote_target("openai/gpt-4.1")
    kwargs = client._build_remote_kwargs(
        or_target,
        messages,
        "medium",
        512,
        "auto",
        None,
        [{
            "type": "function",
            "function": {"name": "alpha_tool", "description": "a", "parameters": {"type": "object"}},
            "cache_control": {"type": "ephemeral"},
        }],
    )

    assert kwargs["messages"][0]["content"] == "cached result"
    assert "cache_control" not in kwargs["messages"][1]["content"][0]
    assert "cache_control" not in kwargs["tools"][0]
    # An unsupported OpenRouter family gains no marker at the send-time finalizer either.
    assert client._normalize_payload_cache_ttl(or_target, kwargs) is None
    assert "cache_control" not in kwargs["tools"][0]
    assert "cache_control" in messages[0]["content"][0]
    assert "cache_control" in messages[1]["content"][0]

    kwargs = client._build_remote_kwargs(
        or_target,
        [{"role": "user", "content": "hi"}],
        "medium",
        512,
        "auto",
        None,
        [{
            "type": "function",
            "function": {
                "name": "schema_tool",
                "description": "schema property collision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cache_control": {"type": "string"},
                    },
                },
            },
        }],
    )
    assert client._normalize_payload_cache_ttl(or_target, kwargs) is None


def test_build_anthropic_tools_deduplicates_tool_names():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dup_tool",
                "description": "first",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dup_tool",
                "description": "second",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    anthropic_tools = LLMClient._build_anthropic_tools(tools)

    assert anthropic_tools == [
        {
            "name": "dup_tool",
            "description": "first",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    sorted_tools = [
        {"type": "function", "function": {"name": "zeta_tool", "description": "z", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "alpha_tool", "description": "a", "parameters": {"type": "object"}}},
    ]

    anthropic_tools = LLMClient._build_anthropic_tools(sorted_tools)

    assert [tool["name"] for tool in anthropic_tools] == ["alpha_tool", "zeta_tool"]
    # v6.77.0: the builder no longer marks anything — the send-time payload finalizer is
    # the single home for cache markers (end-to-end lock:
    # test_chat_anthropic_sends_tool_cache_control_without_ttl).
    assert all("cache_control" not in tool for tool in anthropic_tools)


def test_chat_anthropic_sends_tool_cache_control_without_ttl(monkeypatch):
    from types import SimpleNamespace
    import requests

    # Pinned under the explicit 'default' global TTL: this golden is about the
    # finalizer's marker placement, not the global-override stamping (which has
    # its own goldens in test_review_prompt_caching.py).
    monkeypatch.setenv("OUROBOROS_PROMPT_CACHE_TTL", "default")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    captured = {}
    fake_response = SimpleNamespace(
        raise_for_status=lambda: None,
        status_code=200,
        json=lambda: {"content": [{"type": "text", "text": "ok"}], "usage": {}},
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda _url, headers=None, json=None, timeout=None: (
            captured.update({"payload": json}) or fake_response
        ),
    )
    client = LLMClient()
    message, usage = client._chat_anthropic(
        client._resolve_remote_target("anthropic::claude-sonnet-4.6"),
        [{"role": "user", "content": "hi"}],
        [
            {"type": "function", "function": {"name": "zeta_tool", "description": "z", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "alpha_tool", "description": "a", "parameters": {"type": "object"}}},
        ],
        "medium",
        128,
        "auto",
    )

    assert message["content"] == "ok"
    assert [tool["name"] for tool in captured["payload"]["tools"]] == ["alpha_tool", "zeta_tool"]
    assert captured["payload"]["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "ttl" not in captured["payload"]["tools"][-1]["cache_control"]
    assert usage["prompt_cache_ttl"] == "default"


def test_resolve_anthropic_target_normalizes_direct_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    target = LLMClient()._resolve_remote_target("anthropic::claude-sonnet-4.6")

    assert target["provider"] == "anthropic"
    assert target["resolved_model"] == "claude-sonnet-4-6"
    assert target["usage_model"] == "anthropic/claude-sonnet-4-6"
    assert target["base_url"] == "https://api.anthropic.com/v1"


def test_normalize_anthropic_response_maps_tool_use(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    client = LLMClient()
    target = client._resolve_remote_target("anthropic::claude-sonnet-4-6")
    message, usage = client._normalize_anthropic_response(
        {
            "content": [
                {"type": "text", "text": "Working on it."},
                {"type": "tool_use", "id": "toolu_1", "name": "echo_tool", "input": {"text": "hello"}},
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
        },
        target,
    )

    assert message["content"] == "Working on it."
    assert message["tool_calls"][0]["function"]["name"] == "echo_tool"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"text": "hello"}'
    assert usage["provider"] == "anthropic"
    assert usage["resolved_model"] == "anthropic/claude-sonnet-4-6"
    assert usage["cached_tokens"] == 3
    assert usage["cache_write_tokens"] == 2
    # v6.77.0: Anthropic EXCLUDES cache reads/writes from input_tokens, while every
    # prompt_tokens consumer (pricing's regular_input subtraction, cache_hit_rate, the
    # route calibration ratio) assumes the OpenAI-semantics TOTAL input.
    assert usage["prompt_tokens"] == 11 + 3 + 2
    assert usage["prompt_tokens"] - usage["cached_tokens"] - usage["cache_write_tokens"] == 11
    assert usage.get("cost") is None
    assert usage.get("cost_estimated") is None
    assert usage.get("cost_final") is False


def test_build_anthropic_messages_preserves_system_blocks_and_cache_control(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    client = LLMClient()
    system_blocks, anthropic_messages = client._build_anthropic_messages([
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "dynamic"},
            ],
        },
        {"role": "user", "content": "hi"},
    ])

    assert system_blocks == [
        # v6.69.0: a provider-valid caller ttl survives on the direct
        # Anthropic lane (the 1h tier is GA on the Messages API).
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": "dynamic"},
    ]
    assert anthropic_messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_resolve_openai_compatible_target_prefers_dedicated_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "compat-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compat.example/v1")

    target = LLMClient()._resolve_remote_target("openai-compatible::meta-llama/compatible")

    assert target["provider"] == "openai-compatible"
    assert target["api_key"] == "compat-key"
    assert target["base_url"] == "https://compat.example/v1"
    assert target["usage_model"] == "openai-compatible/meta-llama/compatible"


def test_resolve_cloudru_target_uses_default_base_url(monkeypatch):
    monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "cloudru-key")
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_BASE_URL", raising=False)

    target = LLMClient()._resolve_remote_target("cloudru::giga-model")

    assert target["provider"] == "cloudru"
    assert target["api_key"] == "cloudru-key"
    assert target["base_url"] == "https://foundation-models.api.cloud.ru/v1"
    assert target["usage_model"] == "cloudru/giga-model"


@pytest.mark.parametrize(("region", "expected_base_url"), [
    ("", "https://api.minimax.io/v1"),
    ("global_en", "https://api.minimax.io/v1"),
    ("CN_ZH", "https://api.minimaxi.com/v1"),
])
def test_resolve_minimax_target_uses_regional_endpoint(monkeypatch, region, expected_base_url):
    monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
    if region:
        monkeypatch.setenv("MINIMAX_REGION", region)
    else:
        monkeypatch.delenv("MINIMAX_REGION", raising=False)

    target = LLMClient()._resolve_remote_target("minimax::MiniMax-M3")

    assert target["provider"] == "minimax"
    assert target["resolved_model"] == "MiniMax-M3"
    assert target["usage_model"] == "minimax/MiniMax-M3"
    assert target["api_key"] == "minimax-key"
    assert target["base_url"] == expected_base_url
    assert target["supports_openrouter_extensions"] is False


def test_resolve_gigachat_target_uses_defaults(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "giga-creds")
    monkeypatch.delenv("GIGACHAT_SCOPE", raising=False)
    monkeypatch.delenv("GIGACHAT_BASE_URL", raising=False)
    monkeypatch.delenv("GIGACHAT_VERIFY_SSL_CERTS", raising=False)

    target = LLMClient()._resolve_remote_target("gigachat::GigaChat-2-Max")

    assert target["provider"] == "gigachat"
    assert target["api_key"] == "giga-creds"
    assert target["scope"] == "GIGACHAT_API_PERS"
    assert target["base_url"] == "https://api.giga.chat/v1"
    assert target["verify_ssl_certs"] is True
    assert target["usage_model"] == "gigachat/GigaChat-2-Max"


def test_resolve_gigachat_target_honors_overrides(monkeypatch):
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "giga-creds")
    monkeypatch.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://giga.example/api/v1")
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "false")

    target = LLMClient()._resolve_remote_target("gigachat::GigaChat")

    assert target["scope"] == "GIGACHAT_API_CORP"
    assert target["base_url"] == "https://giga.example/api/v1"
    assert target["verify_ssl_certs"] is False


def test_normalize_remote_response_keeps_direct_openai_cost_unknown(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")
    message, usage = client._normalize_remote_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
            },
        },
        target,
    )

    assert message["content"] == "ok"
    assert usage["provider"] == "openai"
    assert usage["resolved_model"] == "openai/gpt-5.2"
    assert usage["cached_tokens"] == 10
    assert usage["cache_write_tokens"] == 5
    assert usage.get("cost") is None
    assert usage.get("cost_estimated") is None
    assert usage.get("cost_final") is False


def test_normalize_remote_response_preserves_reasoning_and_response_id():
    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")
    message, usage = client._normalize_remote_response(
        {
            "id": "gen-123",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
                    "reasoning": "look up the file",
                    "reasoning_details": [{"type": "reasoning.text", "text": "look up the file"}],
                },
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "response_finish_reason": "provider-fake",
                "response_provider": "provider-secret",
            },
        },
        target,
        skip_cost_fetch=True,
    )

    assert message["response_id"] == "gen-123"
    assert message["reasoning"] == "look up the file"
    assert message["reasoning_details"] == [{"type": "reasoning.text", "text": "look up the file"}]
    assert usage["provider"] == "openrouter"
    assert "response_finish_reason" not in usage
    assert "response_provider" not in usage


def test_build_anthropic_messages_rejects_tool_result_without_tool_call_id(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    client = LLMClient()

    with pytest.raises(ValueError, match="tool_call_id"):
        client._build_anthropic_messages([
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "done"},
        ])


def test_body_error_reroute_strips_reasoning_for_openai_family():
    """Field 2026-07: OpenAI encrypted-reasoning items are NOT reliably portable
    across OpenRouter sibling upstreams (gpt-5.6-sol on 3x OpenAI + 2x Azure:
    'The encrypted content for item rs_... could not be ...' 400s after
    429-reroutes killed benchmark waves). The transient reroute therefore strips
    for openai/* as it did before v6.49.0; Anthropic/Gemini preserve is pinned by
    test_body_error_reroute_preserves_reasoning_for_portable_family."""
    inst = LLMClient.__new__(LLMClient)
    target = {"supports_openrouter_extensions": True}
    kwargs = {
        "model": "openai/gpt-5.6-sol",
        "messages": [
            {"role": "assistant", "reasoning_details": [{"type": "reasoning.text", "signature": "s"}]},
            {"role": "user", "content": "hi"},
        ],
        "extra_body": {"provider": {}},
    }
    rerouted = inst._reroute_same_model_kwargs(target, kwargs, allow_portable_reasoning=True)
    assert rerouted is not None
    assert LLMClient._has_replayed_reasoning_metadata(rerouted["messages"]) is False


def _encrypted_400_kwargs():
    return {
        "model": "openai/gpt-5.6-sol",
        "extra_body": {"provider": {}},
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "prior",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "blob"}],
                "response_id": "resp-1",
            },
        ],
    }


def test_encrypted_body_error_400_strips_and_retries_once():
    """An encrypted-reasoning 400 arriving in the BODY of an HTTP-200 gets the same
    one-shot strip-and-retry as the exception path, instead of surfacing as a
    permanent bad_request that kills the task (body-error hole open since v6.36.0)."""
    client = LLMClient()
    target = client._resolve_remote_target("openai/gpt-5.6-sol")
    calls = []

    class _ErrResp:
        def model_dump(self):
            return {"error": {"code": 400, "message": (
                "The encrypted content for item rs_abc123 could not be verified"
            )}, "usage": {}}

    class _OkResp:
        def model_dump(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def fake_create(**call_kwargs):
        calls.append(call_kwargs)
        return _ErrResp() if len(calls) == 1 else _OkResp()

    resp = client._create_chat_completion_with_retries(
        fake_create, _encrypted_400_kwargs(), target
    )
    assert resp.model_dump()["choices"][0]["message"]["content"] == "ok"
    assert len(calls) == 2
    assert LLMClient._has_replayed_reasoning_metadata(calls[1]["messages"]) is False


def test_genuine_400_body_error_is_not_strip_retried():
    """A non-encrypted bad_request body error must NOT trigger the strip-retry:
    it stays a single send whose typed permanent classification is preserved."""
    client = LLMClient()
    target = client._resolve_remote_target("openai/gpt-5.6-sol")
    calls = []

    class _ErrResp:
        def model_dump(self):
            return {"error": {"code": 400, "message": "Invalid request: bad tool schema"}, "usage": {}}

    def fake_create(**call_kwargs):
        calls.append(call_kwargs)
        return _ErrResp()

    resp = client._create_chat_completion_with_retries(
        fake_create, _encrypted_400_kwargs(), target
    )
    assert resp.model_dump().get("error", {}).get("code") == 400
    assert len(calls) == 1


# --- issue #468: shape-first sealed-reasoning classification -------------------

def _sealed(details=None, content=None, model="z-ai/glm-4.6", **extra):
    from ouroboros.reasoning_artifacts import transcript_has_sealed_reasoning

    msg = {"role": "assistant", **extra}
    if details is not None:
        msg["reasoning_details"] = details
    if content is not None:
        msg["content"] = content
    return transcript_has_sealed_reasoning([{"role": "user", "content": "q"}, msg], model)


@pytest.mark.parametrize("kwargs,expected", [
    # --- portable for EVERY family: readable text carries no endpoint binding ---
    ({"details": [{"type": "reasoning.text", "text": "t"}]}, False),
    ({"details": [{"type": "reasoning.summary", "summary": "s"}]}, False),
    # A sidecar ``data`` on a READABLE type must not seal: opaqueness is a property
    # of the TYPE. Reading it as "has data => opaque" is how #468 comes back dressed up.
    ({"details": [{"type": "reasoning.text", "text": "t", "data": "x"}]}, False),
    # ``None``/``""`` is how providers spell "unsigned" on a readable artifact.
    ({"details": [{"type": "reasoning.text", "text": "t", "signature": None}]}, False),
    ({"details": [{"type": "reasoning.text", "text": "t", "signature": ""}]}, False),
    ({"details": [{"type": "reasoning.text", "text": "t", "signature": "  "}]}, False),
    ({"content": [{"type": "thinking", "thinking": "p"}]}, False),
    ({"content": [{"type": "reasoning", "text": "p"}]}, False),
    ({"reasoning": "plain chain of thought"}, False),
    ({"reasoning_content": "plain chain of thought"}, False),
    ({"response_id": "gen-1"}, False),  # an identifier is not an artifact (Q3)
    ({}, False),
    # --- sealed for a non-roster family ---
    ({"details": [{"type": "reasoning.text", "text": "t", "signature": "sig"}]}, True),
    # A signed SUMMARY seals too: a signature is an endpoint binding wherever it
    # rides on a readable entry (fail-closed; roast finding F6).
    ({"details": [{"type": "reasoning.summary", "summary": "s", "signature": "sig"}]}, True),
    # Non-string truthy flat reasoning fields are unrecognized shapes => fail-closed.
    ({"reasoning": {"weird": "dict"}}, True),
    ({"reasoning_content": ["weird", "list"]}, True),
    # A truthy NON-STRING signature is an unrecognized shape => fail-closed too.
    ({"details": [{"type": "reasoning.text", "text": "t", "signature": 123}]}, True),
    ({"content": [{"type": "thinking", "thinking": "p", "signature": b"sig"}]}, True),
    # FALSY-present reasoning_details is the common JSON idiom for "no reasoning at
    # all" — NOT an artifact. Sealing it would pin transcripts with zero reasoning
    # (the response_id bug class); parity with the presence predicate's truthiness.
    ({"details": {}}, False),
    ({"details": ""}, False),
    ({"details": 0}, False),
    ({"details": False}, False),
    ({"reasoning_details": None}, False),  # explicit present-null via **extra
    ({"details": [{"type": "reasoning.encrypted", "data": "blob"}]}, True),
    ({"details": [{"type": "reasoning.brand_new_shape"}]}, True),  # unknown => fail-closed
    ({"details": [{"no_type": 1}]}, True),
    ({"details": "not-a-list"}, True),                             # malformed => fail-closed
    ({"details": {"type": "reasoning.text"}}, True),
    ({"content": [{"type": "redacted_thinking", "data": "blob"}]}, True),
    ({"content": [{"type": "thinking", "thinking": "p", "signature": "sig"}]}, True),
    ({"content": [{"type": "text", "text": "hi", "signature": "stray"}]}, True),
])
def test_sealed_classifier_single_forms(kwargs, expected):
    """One artifact form at a time, on a NON-ROSTER family, so the verdict is the
    shape's alone."""
    assert _sealed(**kwargs) is expected


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4.6", "google/gemini-3.5-flash",
                                   "~google/gemini-3.5-flash", " anthropic/claude-opus-5 ",
                                   "Anthropic/Claude-Opus-5"])
@pytest.mark.parametrize("form", [
    {"details": [{"type": "reasoning.encrypted", "data": "blob"}]},
    {"details": [{"type": "reasoning.text", "text": "t", "signature": "sig"}]},
    {"details": [{"type": "reasoning.unknown_future"}]},
    {"content": [{"type": "redacted_thinking", "data": "blob"}]},
])
def test_signed_portable_roster_vouches_every_opaque_form(model, form):
    """The roster is the SECOND axis and it vouches ALL non-plain forms for its
    families — exactly the exemption they held before the classifier, so their
    same-model failover behavior is byte-for-byte unchanged. Model ids are
    normalized (leading ``~``, surrounding whitespace) before the prefix test."""
    assert _sealed(model=model, **form) is False


def test_sealed_classifier_ignores_malformed_transcript_entries():
    from ouroboros.reasoning_artifacts import transcript_has_sealed_reasoning

    assert transcript_has_sealed_reasoning([], "z-ai/glm-4.6") is False
    assert transcript_has_sealed_reasoning(["not-a-dict", None], "z-ai/glm-4.6") is False
    assert transcript_has_sealed_reasoning(
        ["not-a-dict", {"reasoning_details": [{"type": "reasoning.encrypted"}]}],
        "z-ai/glm-4.6",
    ) is True


def test_sealed_artifact_label_is_a_bounded_vocabulary():
    """The label rides into usage telemetry, so it must come from a closed host-owned
    set — never provider-controlled text."""
    from ouroboros.reasoning_artifacts import sealed_reasoning_artifact

    cases = {
        "encrypted": [{"type": "reasoning.encrypted", "data": "b"}],
        "signed_text": [{"type": "reasoning.text", "signature": "s"}],
        "unknown_type": [{"type": "reasoning.mystery"}],
        "reasoning_details_malformed": "not-a-list",
    }
    for label, details in cases.items():
        assert sealed_reasoning_artifact(
            [{"role": "assistant", "reasoning_details": details}], "z-ai/glm-4.6",
        ) == label
    assert sealed_reasoning_artifact(
        [{"role": "assistant", "content": [{"type": "redacted_thinking"}]}], "z-ai/glm-4.6",
    ) == "redacted_thinking"


# --- issue #468: gap-merge against the owner provider policy -------------------

@pytest.mark.parametrize("details,expect_pin", [
    ([{"type": "reasoning.encrypted", "data": "blob"}], True),
    ([{"type": "reasoning.text", "text": "t"}], False),
])
def test_owner_allow_fallbacks_policy_loses_only_to_a_sealed_pin(monkeypatch, details, expect_pin):
    """The owner's ``OUROBOROS_OR_PROVIDER`` resilience policy is honored unless the
    continuity pin would be silently overwritten into a GUARANTEED 400. A portable
    transcript has no such pin, so ``allow_fallbacks:true`` must reach the wire — that
    is the exact configuration issue #468 was filed against."""
    monkeypatch.setenv(
        "OUROBOROS_OR_PROVIDER", '{"allow_fallbacks": true, "require_parameters": true}',
    )
    _kwargs, provider = _dispatch_provider_block(
        monkeypatch, "deepseek/deepseek-v4-flash-0731",
        _replay_assistant_turn(reasoning_details=details),
    )

    assert provider["require_parameters"] is True
    assert provider["allow_fallbacks"] is (False if expect_pin else True)


# --- issue #468: pin telemetry ------------------------------------------------

def _pin_usage(model, messages, monkeypatch):
    """Drive a real send through the retry ladder and return the normalized usage."""
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    target = client._resolve_remote_target(model)
    kwargs = client._build_remote_kwargs(
        target, messages, "medium", 512, "auto", None, None,
    )

    class _Resp:
        def model_dump(self):
            return {
                "id": "gen-1", "provider": "Fireworks",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          # A provider usage extension must not be able to spoof the
                          # host-owned pin fact.
                          "reasoning_pin": {"sealed": False, "artifact": "spoofed"}},
            }

    resp = client._create_chat_completion_with_retries(lambda **_kw: _Resp(), kwargs, target)
    _msg, usage = client._normalize_remote_response(
        resp.model_dump(), target, skip_cost_fetch=True,
    )
    return usage


def test_sealed_pin_is_disclosed_in_usage(monkeypatch):
    """Why same-model failover was withheld must be recoverable from the call's own
    usage record, not by excavating observability blobs (issue #468)."""
    usage = _pin_usage(
        "z-ai/glm-4.6",
        _replay_assistant_turn(
            reasoning_details=[{"type": "reasoning.encrypted", "data": "blob"}],
        ),
        monkeypatch,
    )

    assert usage["reasoning_pin"] == {"sealed": True, "artifact": "encrypted"}
    assert usage["response_provider"] == "Fireworks"


def test_portable_transcript_discloses_no_pin(monkeypatch):
    usage = _pin_usage(
        "z-ai/glm-4.6",
        _replay_assistant_turn(reasoning_details=[{"type": "reasoning.text", "text": "t"}]),
        monkeypatch,
    )

    # The spoofed provider-supplied key is popped and NOT replaced: no pin, no fact.
    assert "reasoning_pin" not in usage


def test_pin_fact_follows_the_terminal_candidate_not_the_built_one(monkeypatch):
    """The recovery ladder can strip reasoning and drop the pin. The disclosure is
    staged on send SUCCESS, so it describes what actually reached the wire — a
    build-time guess would report a pin the terminal request never carried."""
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _model_id: None)
    target = client._resolve_remote_target("z-ai/glm-4.6")
    kwargs = client._build_remote_kwargs(
        target,
        _replay_assistant_turn(
            reasoning_details=[{"type": "reasoning.encrypted", "data": "blob"}],
        ),
        "medium", 512, "auto", None, None,
    )
    assert kwargs["extra_body"]["provider"]["allow_fallbacks"] is False

    sent = []

    class _Resp:
        def __init__(self, dump):
            self._dump = dump

        def model_dump(self):
            return self._dump

    rate_limited = {"error": {"code": 429, "message": "rate"}, "choices": None}
    healthy = {"id": "gen-2", "provider": "Nebius",
               "choices": [{"message": {"content": "ok"}}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def fake_create(**call_kwargs):
        sent.append(call_kwargs)
        return _Resp(rate_limited if len(sent) == 1 else healthy)

    resp = client._create_chat_completion_with_retries(fake_create, kwargs, target)
    _msg, usage = client._normalize_remote_response(
        resp.model_dump(), target, skip_cost_fetch=True,
    )

    # The reroute stripped the sealed artifact and unpinned; the terminal candidate
    # carried no pin, so no pin is disclosed.
    assert len(sent) == 2
    assert "allow_fallbacks" not in sent[-1].get("extra_body", {}).get("provider", {})
    assert "reasoning_pin" not in usage
