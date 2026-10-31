"""v6.37.0 regression guards (C1.1/C1.2): cross-FAMILY model switches must not
replay provider-private reasoning/thinking blocks to a different upstream family.

The forensic failure: GLM (z-ai) reasoning + Anthropic-shaped ``thinking`` content
blocks were replayed verbatim into the ``anthropic/claude-sonnet-4.6`` fallback,
which 400'd with "Invalid `signature` in `thinking` block" — and the old recovery
keyed on an error-STRING allowlist that did not contain that phrase, so it never
fired. These guards lock in the structural fix."""

import pytest


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


def test_signature_retry_leaves_a_textual_context_overflow_to_size_recovery():
    """An OpenRouter 400 whose upstream says the prompt is too long goes to the loop's size
    recovery (the same shared classifier), not to a reasoning strip that history would keep."""
    from ouroboros.llm import LLMClient

    class _BadReq(Exception):
        status_code = 400

    def routed(raw: str, previous: str = "") -> _BadReq:
        nested = f", 'previous_errors': [{{'raw': '{previous}'}}]" if previous else ""
        return _BadReq("Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, "
                       f"'metadata': {{'raw': '{raw}'{nested}}}}}}}")

    inst = LLMClient.__new__(LLMClient)
    target = {"supports_openrouter_extensions": True}
    kwargs = {"messages": _thinking_history(), "extra_body": {"provider": {"allow_fallbacks": False}}}
    too_long = "prompt is too long: 1000794 tokens > 1000000 maximum"
    bound = "messages.3.content.0: Invalid `signature` in `thinking` block: bound to a different conversation"
    from ouroboros.loop_llm_call import classify_llm_exception

    for overflow in (routed(too_long), routed(bound, too_long)):
        assert inst._openrouter_signature_retry_kwargs(target, kwargs, overflow) is None
        assert classify_llm_exception(overflow).kind == "context_overflow"  # the loop takes it over
    assert inst._openrouter_signature_retry_kwargs(target, kwargs, routed(bound)) is not None


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


def _direct_lane_payload(monkeypatch, verdict):
    """A direct (non-OpenRouter) OpenAI-lane send: the one send policy, then the builder."""
    import json as _json
    from ouroboros import vision_routing
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setattr(vision_routing, "supports_vision", lambda *_a, **_k: verdict)
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *_a, **_k: "")
    sent = vision_routing.prepare_messages_for_send(_image_history(), routing=vision_routing.VisionRoutingContext(
        "openai::some-direct-model", object(), {}))
    target = {"resolved_model": "some-direct-model", "provider": "openai",
              "supports_openrouter_extensions": False}
    kwargs = LLMClient()._build_remote_kwargs(
        target, sent, reasoning_effort="low", max_tokens=64,
        tool_choice="auto", temperature=None, tools=None,
    )
    return _json.dumps(kwargs["messages"])


def test_confirmed_text_only_route_gets_no_pixels_on_the_direct_lane(monkeypatch):
    """C2.3, moved to the policy: a route whose own metadata says it cannot see gets
    an honest marker on the DIRECT lane too (Auto, no caption route); the builder
    itself only encodes what the policy gave it."""
    blob = _direct_lane_payload(monkeypatch, False)
    assert "image_url" not in blob  # raw image block was replaced
    assert "base64,AAAA" not in blob  # the image payload is gone
    assert "image omitted" in blob


def test_unknown_or_seeing_route_keeps_image_on_direct_lane(monkeypatch):
    """Unknown is not no: an unknown route and a confirmed one both keep the image."""
    for verdict in (True, None):
        assert "image_url" in _direct_lane_payload(monkeypatch, verdict)

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


def _signed_turn_kwargs(client, target):
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
         "reasoning_details": [{"type": "reasoning.text", "text": "t", "signature": "s",
                                "format": "anthropic-claude-v1", "index": 0}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]
    return client._build_remote_kwargs(target, history, "high", 128, "auto", None, None, skip_capability_fetch=True)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("scenario", ["answered", "refused_then_answered", "refused_twice", "refused_then_error_body"])
def test_only_an_answered_send_without_the_replayed_reasoning_notes_the_omission(monkeypatch, asynchronous, scenario):
    """The omission fact describes the send whose generated answer came back: the stripped
    retry after a 400 notes it; an untouched success, a retry answered only with an error
    body (the model read nothing) and a ladder that ends in an exception note nothing."""
    import asyncio
    from ouroboros.llm import LLMClient
    from ouroboros.reasoning_artifacts import note_replay_omission, pop_replay_omission

    class _BadReq(Exception):
        status_code = 400

    class _Resp:
        def __init__(self, body):
            self.body = body

        def model_dump(self):
            return self.body

    answer = {"id": "gen-1", "provider": "P", "usage": {"prompt_tokens": 1},
              "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    error_body = {"id": "gen-2", "error": {"code": 502, "message": "upstream unavailable"}, "choices": None}
    plan = {"answered": [answer], "refused_then_answered": [None, answer], "refused_twice": [None, None],
            "refused_then_error_body": [None, error_body]}[scenario]
    client = LLMClient()
    monkeypatch.setattr(client, "_get_supported_parameters", lambda _m: None)
    target = client._resolve_remote_target("anthropic/claude-opus-5.5")
    sent = []

    def create(**candidate):
        sent.append(candidate["messages"])
        reply = plan[min(len(sent), len(plan)) - 1]
        if reply is None:
            raise _BadReq("Error code: 400 - Invalid `signature` in `thinking` block")
        return _Resp(reply)

    async def create_async(**candidate):
        return create(**candidate)

    def observe(outcome):  # the fact lives in the caller's context: read it there
        return outcome, pop_replay_omission()

    async def run_async():
        note_replay_omission(True)  # residue of an earlier call must not survive the ladder's entry
        try:
            return observe(await client._create_chat_completion_with_retries_async(
                create_async, _signed_turn_kwargs(client, target), target))
        except _BadReq as error:
            return observe(error)

    def run_sync():
        note_replay_omission(True)
        try:
            return observe(client._create_chat_completion_with_retries(create, _signed_turn_kwargs(client, target), target))
        except _BadReq as error:
            return observe(error)

    outcome, omitted = asyncio.run(run_async()) if asynchronous else run_sync()
    assert isinstance(outcome, _BadReq) is (scenario == "refused_twice")
    assert omitted is (scenario == "refused_then_answered")
    assert LLMClient._has_replayed_reasoning_metadata(sent[-1]) is (scenario == "answered")
