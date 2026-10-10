"""Existing cache boundaries reach OpenRouter without changing canonical messages."""
import copy
import json

import httpx
import openai
import pytest

from ouroboros.llm import LLMClient


@pytest.mark.parametrize("provider", ["openrouter", "openai", "openai-compatible"])
def test_sdk_wire_keeps_boundaries_only_on_supported_transport(provider):
    original = [{"role": "system", "content": [
        {"type": "text", "text": "Stable identity and memory", "_caption": "host only",
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"type": "text", "text": " ", "cache_control": {"type": "ephemeral"}},
    ]}, {"role": "user", "content": "Current task"}]
    before = copy.deepcopy(original)
    target = {"provider": provider, "resolved_model": "openai/gpt-6.1-sol",
              "usage_model": "openai/gpt-6.1-sol",
              "supports_openrouter_extensions": provider == "openrouter"}
    client = LLMClient(api_key="fixture")
    payload = client._build_remote_kwargs(target, original, "low", 128, "auto", None, None,
                                         skip_capability_fetch=True)
    client._normalize_payload_cache_ttl(target, payload)
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-6.1-sol", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "done"}}]})

    with openai.OpenAI(api_key="fixture", base_url="https://fixture.invalid/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond))) as sdk:
        sdk.chat.completions.create(**payload)
    wire = json.dumps(sent[0], ensure_ascii=False)
    assert "Stable identity and memory" in wire and "Current task" in wire
    assert "_caption" not in wire and "ttl" not in wire
    if provider == "openrouter":
        blocks = sent[0]["messages"][0]["content"]
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in blocks[1]
    else:
        assert "cache_control" not in wire
    assert original == before
