"""Real image producers, eviction, and serialized Chat/Anthropic send candidates."""
from __future__ import annotations

import base64
import copy
import io
import json
import os

import pytest
from PIL import Image

from ouroboros.llm import LLMClient
from ouroboros.loop_messages import _evict_stale_image_blocks
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_image_ordering_normalize import _assistant_toolcall
from tests.test_tool_result_delivery import _row
from ouroboros.loop_tool_execution import process_tool_results


def _wire(messages, provider):
    target = {"provider": provider, "resolved_model": "fixture-model", "supports_openrouter_extensions": False}
    payload = LLMClient()._build_remote_candidate(target, messages, "none", 1000, "auto", None, [])
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))


@pytest.mark.parametrize("producer", ["view_image", "browser"])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_completed_episode_keeps_tool_before_image_after_real_eviction(tmp_path, monkeypatch, producer, provider,
                                                                     record_property):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    repo = tmp_path / "repo"
    repo.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (20, 50, 100)).save(buffer, format="PNG")
    source = repo / "image.png"
    source.write_bytes(buffer.getvalue())
    messages = [{"role": "user", "content": "Inspect the image."}, _assistant_toolcall()]
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="image-reader", messages=messages)
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path)
    registry.set_context(ctx)
    if producer == "view_image":
        result = registry.execute("view_image", {"path": str(source)})
        assert "attached as a local image" in result, result
    else:
        from ouroboros.tools.browser import _inject_native_screenshot
        result = _inject_native_screenshot(ctx, base64.b64encode(buffer.getvalue()).decode())
        assert result
    process_tool_results([_row("c1", "view_image", result)], messages, {"tool_calls": []},
                         lambda *_a, **_k: None, registry)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "tool"]
    original = copy.deepcopy(messages)
    before = _wire(messages, provider)
    assert messages == original  # serializers never repair canonical history
    from ouroboros import vision_routing as vision
    usage = {}
    monkeypatch.setattr(vision, "image_route_key", lambda *_a, **_k: "fixture-route")
    for mode in ("caption", "off", "blind", "refusal"):
        monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline" if mode == "refusal" else
                           "auto" if mode == "blind" else mode)
        if mode == "refusal":
            digests = vision.query_image_digests([{"url": original[2]["content"][1]["image_url"]["url"]}])
            vision.record_image_refusal(usage, "fixture-model", digests,
                                        {"status": 400, "code": "image_rejected", "message": "fixture refusal"})
        projected = vision.prepare_messages_for_send(original, routing=vision.VisionRoutingContext(
            "fixture-model", object(), usage, use_local=mode == "blind"))
        assert original == copy.deepcopy(messages)
        assert all(block["type"] == "text" for block in projected[2]["content"])
        payload = _wire(projected, provider)
        answer = payload["messages"][2]
        if provider == "openai":
            assert answer["role"] == "tool" and answer["content"] == result
        else:
            assert answer["content"][0]["type"] == "tool_result"
            assert answer["content"][0]["content"] == result
        record_property(mode + "_payload", json.dumps(payload, ensure_ascii=False, sort_keys=True))
    from ouroboros.context_budget import LIVE_IMAGE_BLOCKS_HIGH_WATER
    assert _evict_stale_image_blocks(messages, incoming=LIVE_IMAGE_BLOCKS_HIGH_WATER)
    assert messages[-1] == original[-1] and messages[-1]["content"] == result
    evicted = copy.deepcopy(messages)
    after = _wire(messages, provider)
    assert messages == evicted
    before_json, after_json = (json.dumps(p, ensure_ascii=False, sort_keys=True) for p in (before, after))
    record_property("before_payload", before_json)
    record_property("after_payload", after_json)
    record_property("first_changed_character", len(os.path.commonprefix([before_json, after_json])))
    if provider == "openai":
        assert [m["role"] for m in before["messages"]] == ["user", "assistant", "tool", "user"]
        assert [m["role"] for m in after["messages"]] == ["user", "assistant", "tool", "user"], after
        assert before["messages"][:3] == after["messages"][:3]
        assert after["messages"][2]["content"] == result
    else:
        for payload in (before, after):
            content = payload["messages"][2]["content"]
            assert content[0]["type"] == "tool_result", payload
            assert content[0]["tool_use_id"] == "c1" and content[0]["content"] == result
        assert before["messages"][:2] == after["messages"][:2]
