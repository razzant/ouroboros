"""v6.37.0 regression guards (C1.1/C1.2): cross-FAMILY model switches must not
replay provider-private reasoning/thinking blocks to a different upstream family.

The forensic failure: GLM (z-ai) reasoning + Anthropic-shaped ``thinking`` content
blocks were replayed verbatim into the ``anthropic/claude-sonnet-4.6`` fallback,
which 400'd with "Invalid `signature` in `thinking` block" — and the old recovery
keyed on an error-STRING allowlist that did not contain that phrase, so it never
fired. These guards lock in the structural fix."""


def _thinking_history():
    return [
        {
            "role": "assistant",
            "reasoning_details": [{"type": "reasoning.text", "text": "glm thoughts"}],
            "response_id": "resp_123",
            "content": [
                {"type": "thinking", "thinking": "secret chain", "signature": "glm-sig-xyz"},
                {"type": "text", "text": "the answer"},
            ],
        },
        {"role": "user", "content": "hi"},
    ]


def test_model_family_boundary():
    from ouroboros.llm import LLMClient
    assert LLMClient._model_family("z-ai/glm-5.2") == "z-ai"
    assert LLMClient._model_family("anthropic/claude-sonnet-4.6") == "anthropic"
    assert LLMClient._model_family("openai/gpt-5.5") == "openai"
    # bare/local id -> itself (still a usable boundary)
    assert LLMClient._model_family("local-model") == "local-model"


def test_has_replayed_reasoning_metadata_detects_all_shapes():
    from ouroboros.llm import LLMClient
    assert LLMClient._has_replayed_reasoning_metadata(_thinking_history()) is True
    assert LLMClient._has_replayed_reasoning_metadata(
        [{"role": "assistant", "reasoning_details": [{"x": 1}], "content": "p"}]
    ) is True
    assert LLMClient._has_replayed_reasoning_metadata(
        [{"role": "assistant", "content": [{"type": "thinking", "signature": "s"}]}]
    ) is True
    # clean transcript -> no replayed reasoning
    assert LLMClient._has_replayed_reasoning_metadata(
        [{"role": "assistant", "content": "plain"}, {"role": "user", "content": "x"}]
    ) is False


def test_strip_removes_thinking_blocks_and_signatures_keeps_text():
    from ouroboros.llm import LLMClient
    out = LLMClient._strip_openrouter_roundtrip_metadata(_thinking_history())
    assert LLMClient._has_replayed_reasoning_metadata(out) is False
    asst = out[0]
    assert "reasoning_details" not in asst and "response_id" not in asst
    # the thinking block is gone; the real text answer survives
    assert asst["content"] == [{"type": "text", "text": "the answer"}]
    # canonical input is NOT mutated (deep copy)
    assert any(
        isinstance(b, dict) and b.get("type") == "thinking"
        for b in _thinking_history()[0]["content"]
    )


def test_is_http_status_structural():
    from ouroboros.llm import LLMClient

    class _BadReq(Exception):
        status_code = 400

    assert LLMClient._is_http_status(_BadReq("boom"), 400) is True
    assert LLMClient._is_http_status(_BadReq("boom"), 429) is False
    # fallback to OpenAI-SDK message shape when no status_code attribute
    assert LLMClient._is_http_status(Exception("Error code: 400 - bad"), 400) is True
    assert LLMClient._is_http_status(Exception("nope"), 400) is False


def test_signature_retry_fires_on_400_with_thinking_block_no_string_marker():
    """The exact forensic failure: a 400 whose message ('Invalid signature in
    thinking block') matches NONE of the old markers must STILL trigger the
    strip-and-retry, because the trigger is now structural (400 + request carried
    replayed reasoning)."""
    from ouroboros.llm import LLMClient

    class _BadReq(Exception):
        status_code = 400

    inst = LLMClient.__new__(LLMClient)
    target = {"supports_openrouter_extensions": True}
    kwargs = {
        "messages": _thinking_history(),
        "extra_body": {"provider": {"allow_fallbacks": False}},
    }
    exc = _BadReq("messages.1.content.0: Invalid `signature` in `thinking` block")
    out = inst._openrouter_signature_retry_kwargs(target, kwargs, exc)
    assert out is not None
    assert LLMClient._has_replayed_reasoning_metadata(out["messages"]) is False

    # a genuine 400 WITHOUT replayed reasoning is left alone (re-raised upstream)
    clean = {"messages": [{"role": "user", "content": "x"}]}
    assert inst._openrouter_signature_retry_kwargs(target, clean, exc) is None
    # a non-400 error never triggers the reasoning strip
    assert inst._openrouter_signature_retry_kwargs(target, kwargs, Exception("Error code: 429")) is None


def test_sanitize_on_model_switch_strips_cross_family_preserves_same_family():
    from ouroboros.llm import LLMClient
    hist = _thinking_history()
    # cross-family (z-ai -> anthropic): sanitized copy, reasoning gone
    cross = LLMClient.sanitize_reasoning_on_model_switch(hist, "z-ai/glm-5.2", "anthropic/claude-sonnet-4.6")
    assert cross is not hist
    assert LLMClient._has_replayed_reasoning_metadata(cross) is False
    # same family (anthropic -> anthropic): identity, continuity preserved
    same = LLMClient.sanitize_reasoning_on_model_switch(hist, "anthropic/claude-opus-4.8", "anthropic/claude-sonnet-4.6")
    assert same is hist


def _image_history():
    return [
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]


def test_blind_model_placeholder_on_direct_provider_lane(monkeypatch):
    """C2.3 (triad+scope round-2): a BLIND model on the DIRECT (non-OpenRouter)
    OpenAI/Cloud.ru lane must get image placeholders too — the replacement used to
    run only after the direct branch had already returned with raw image blocks."""
    import json as _json
    from ouroboros.llm import LLMClient
    import ouroboros.provider_models as pm

    monkeypatch.setattr(pm, "supports_vision", lambda m: False)
    client = LLMClient()
    target = {"resolved_model": "some/blind-direct-model", "provider": "openai",
              "supports_openrouter_extensions": False}
    kwargs = client._build_remote_kwargs(
        target, _image_history(), reasoning_effort="low", max_tokens=64,
        tool_choice="auto", temperature=None, tools=None,
    )
    blob = _json.dumps(kwargs["messages"])
    assert "image_url" not in blob  # raw image block was replaced
    assert "base64,AAAA" not in blob  # the image payload is gone


def test_vision_model_keeps_image_on_direct_lane(monkeypatch):
    """A vision-capable model on the same direct lane keeps its image block."""
    import json as _json
    from ouroboros.llm import LLMClient
    import ouroboros.provider_models as pm

    monkeypatch.setattr(pm, "supports_vision", lambda m: True)
    client = LLMClient()
    target = {"resolved_model": "some/vision-direct-model", "provider": "openai",
              "supports_openrouter_extensions": False}
    kwargs = client._build_remote_kwargs(
        target, _image_history(), reasoning_effort="low", max_tokens=64,
        tool_choice="auto", temperature=None, tools=None,
    )
    assert "image_url" in _json.dumps(kwargs["messages"])

# --- B1 (v6.39): GLM/cloud.ru OpenAI-compatible top-level reasoning_content ---

def _reasoning_content_history():
    return [
        {
            "role": "assistant",
            "reasoning_content": "glm internal chain of thought",
            "content": "the answer",
        },
        {"role": "user", "content": "next"},
    ]


def test_strip_removes_top_level_reasoning_content():
    from ouroboros.llm import LLMClient
    out = LLMClient._strip_openrouter_roundtrip_metadata(_reasoning_content_history())
    assert "reasoning_content" not in out[0]
    assert out[0]["content"] == "the answer"
    # canonical input not mutated (deep copy)
    assert "reasoning_content" in _reasoning_content_history()[0]


def test_has_replayed_reasoning_metadata_detects_reasoning_content():
    from ouroboros.llm import LLMClient
    assert LLMClient._has_replayed_reasoning_metadata(_reasoning_content_history()) is True


def test_normalize_remote_response_drops_reasoning_content_before_transcript():
    """OFFLINE: GLM/cloud.ru echo their OWN reasoning_content -> strict vLLM 400 on
    the next same-model turn. It must never enter the canonical transcript."""
    from ouroboros.llm import LLMClient
    client = LLMClient(api_key="x")
    resp_dict = {
        "id": "chatcmpl-1",
        "choices": [{"message": {
            "role": "assistant",
            "content": "hello",
            "reasoning_content": "secret glm reasoning",
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    target = {
        "provider": "cloudru",
        "supports_openrouter_extensions": False,
        "supports_generation_cost": False,
        "usage_model": "cloudru/zai-org/GLM-4.7",
    }
    msg, usage = client._normalize_remote_response(resp_dict, target, skip_cost_fetch=True)
    assert "reasoning_content" not in msg
    assert msg.get("content") == "hello"


# --- MiniMax split reasoning: the wire contract, not a parser-level <think> strip ---

def _split_reasoning_history():
    return [
        {
            "role": "assistant",
            "content": "the answer",
            "reasoning": "flat rollup",
            "response_id": "resp_minimax_1",
            "reasoning_details": [{
                "type": "reasoning.text",
                "id": "reasoning-text-1",
                "format": "MiniMax-response-v1",
                "index": 0,
                "text": "minimax chain of thought",
            }],
        },
        {"role": "user", "content": "next"},
    ]


def test_minimax_request_asks_for_split_reasoning():
    """MiniMax wraps its thinking in ``<think>`` tags inside ``content`` unless the
    request carries ``reasoning_split``; it must ride in extra_body because the
    OpenAI SDK rejects unknown top-level kwargs."""
    from ouroboros.llm import LLMClient
    client = LLMClient()
    target = {"resolved_model": "MiniMax-M3", "provider": "minimax",
              "supports_openrouter_extensions": False, "reasoning_split": True}
    kwargs = client._build_remote_kwargs(
        target, [{"role": "user", "content": "hi"}], reasoning_effort="low",
        max_tokens=64, tool_choice="auto", temperature=None, tools=None,
    )
    assert kwargs["extra_body"]["reasoning_split"] is True
    # "reasoning_split" is not a top-level request field on any lane.
    assert "reasoning_split" not in kwargs


def test_lanes_without_split_reasoning_send_no_such_field():
    """Only a target that declares the contract carries it: the DeepSeek and direct
    OpenAI lanes must keep their own wire shape."""
    from ouroboros.llm import LLMClient
    client = LLMClient()
    for target in (
        {"resolved_model": "deepseek-chat", "provider": "deepseek",
         "supports_openrouter_extensions": False, "requires_reasoning_echo": True},
        {"resolved_model": "gpt-5.5", "provider": "openai",
         "supports_openrouter_extensions": False},
    ):
        kwargs = client._build_remote_kwargs(
            target, [{"role": "user", "content": "hi"}], reasoning_effort="low",
            max_tokens=64, tool_choice="auto", temperature=None, tools=None,
        )
        assert "reasoning_split" not in kwargs
        assert "reasoning_split" not in (kwargs.get("extra_body") or {})


def test_split_reasoning_lane_replays_reasoning_details_only():
    """MiniMax's interleaved-thinking contract wants its ``reasoning_details`` records
    back unchanged on the same lane; every OTHER round-trip artifact still goes."""
    from ouroboros.llm import LLMClient
    out = LLMClient._strip_openrouter_roundtrip_metadata(
        _split_reasoning_history(), keep_reasoning_details=True,
    )
    asst = out[0]
    assert asst["reasoning_details"] == _split_reasoning_history()[0]["reasoning_details"]
    assert "reasoning" not in asst and "response_id" not in asst
    assert asst["content"] == "the answer"


def test_default_still_strips_reasoning_details():
    """The default (every other lane, and any cross-family send) is unchanged."""
    from ouroboros.llm import LLMClient
    out = LLMClient._strip_openrouter_roundtrip_metadata(_split_reasoning_history())
    assert "reasoning_details" not in out[0]
    assert "reasoning" not in out[0] and "response_id" not in out[0]
    # a cross-family switch scrubs the split-reasoning transcript like any other
    cross = LLMClient.sanitize_reasoning_on_model_switch(
        _split_reasoning_history(), "minimax/MiniMax-M3", "anthropic/claude-sonnet-4.6",
    )
    assert LLMClient._has_replayed_reasoning_metadata(cross) is False
