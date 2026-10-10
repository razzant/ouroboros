"""VLM route choice and the transport builder under "unknown is not no".

The VLM lane consults route evidence, never a model's name: an explicitly named
model is called even when its metadata says no (its refusal comes back typed), an
automatic choice prefers a confirmed yes, then an unknown, and only a lane where
every candidate is confirmed unable gets the typed ``VLM_NO_VISION_MODEL`` gap.
The transport builder encodes the images it is given on every route.
"""

from types import SimpleNamespace

import pytest

SEES = "google/gemini-3.5-flash"
TEXT_ONLY = "z-ai/glm-5.2"


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """One recorded OpenRouter catalog response in an isolated evidence store."""
    from ouroboros.vision_routing import record_catalog_image_input

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    record_catalog_image_input("openrouter", "https://openrouter.ai/api/v1", [
        {"id": SEES, "architecture": {"input_modalities": ["text", "image"]}},
        {"id": TEXT_ONLY, "architecture": {"input_modalities": ["text"]}},
    ], source="OpenRouter /models")


def test_resolve_vlm_model_follows_route_evidence_not_names(recorded, monkeypatch):
    from ouroboros.tools import vision as V
    client = object()
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "")
    # An explicit model is called even when its metadata says no (owner decision).
    assert V._resolve_vlm_model(client, TEXT_ONLY) == TEXT_ONLY
    assert V._resolve_vlm_model(client, "acme/never-listed-1") == "acme/never-listed-1"

    # No explicit model: a confirmed yes first, then an unknown; a confirmed no is skipped.
    monkeypatch.setattr(
        V, "_vision_capable_slot_candidates",
        lambda c, ctx=None: [TEXT_ONLY, "acme/never-listed-1", SEES],
    )
    assert V._resolve_vlm_model(client, "", ctx=SimpleNamespace()) == SEES
    monkeypatch.setattr(V, "_vision_capable_slot_candidates",
                        lambda c, ctx=None: [TEXT_ONLY, "acme/never-listed-1"])
    assert V._resolve_vlm_model(client, "", ctx=SimpleNamespace()) == "acme/never-listed-1"

    # Every candidate confirmed unable -> "" so the caller surfaces VLM_NO_VISION_MODEL.
    monkeypatch.setattr(V, "_vision_capable_slot_candidates",
                        lambda c, ctx=None: [TEXT_ONLY, "gigachat::GigaChat-2-Max"])
    assert V._resolve_vlm_model(client, "", ctx=SimpleNamespace()) == ""


def test_slot_candidates_prefer_active_then_light_dedup(monkeypatch):
    from ouroboros.tools import vision as V
    monkeypatch.setenv("OUROBOROS_MODEL_HEAVY", "google/gemini-3.5-flash")
    monkeypatch.setenv("OUROBOROS_MODEL", "z-ai/glm-5.2")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "anthropic/claude-sonnet-4.6")
    monkeypatch.setattr("ouroboros.config.get_light_model", lambda: "google/gemini-3.5-flash")

    class _Client:
        def default_model(self):
            return "z-ai/glm-5.2"

    ctx = SimpleNamespace(active_model="x-ai/grok-4", task_model_override="")
    out = V._vision_capable_slot_candidates(_Client(), ctx)
    assert out[0] == "x-ai/grok-4"  # active model leads
    assert "google/gemini-3.5-flash" in out
    assert len(out) == len(set(out))  # de-duplicated, empties dropped


def test_analyze_screenshot_no_vision_names_why_without_forbidding_a_retry(recorded, monkeypatch):
    from ouroboros.tools import vision as V
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "")
    monkeypatch.setattr(V, "_vision_capable_slot_candidates", lambda c, ctx=None: [TEXT_ONLY])
    monkeypatch.setattr(V, "_get_llm_client", lambda: object())
    ctx = SimpleNamespace(browser_state=SimpleNamespace(last_screenshot_b64="aGk="))
    out = V._analyze_screenshot(ctx, prompt="check")
    assert out.startswith("⚠️ VLM_NO_VISION_MODEL:")
    assert TEXT_ONLY in out and "OpenRouter /models" in out  # names the route and its source
    assert "Do NOT retry" not in out


@pytest.mark.parametrize("model", [
    "openai::gpt-5.6-terra", "openai/gpt-5.6-sol", "deepseek/deepseek-chat", "acme/never-listed-1",
])
def test_build_remote_kwargs_encodes_the_images_it_receives(monkeypatch, tmp_path, model):
    """The transport builder never re-judges capability: qualified, bare, listed or
    unlisted, every OpenAI-shaped route carries the image block it is given."""
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    client = LLMClient()
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }]
    target = client._resolve_remote_target(model)
    kwargs = client._build_remote_kwargs(target, msgs, "high", 128, "auto", None, None,
                                         skip_capability_fetch=True)
    assert [b.get("type") for b in kwargs["messages"][0]["content"]] == ["text", "image_url"]


def test_lane_placeholder_names_our_transport_and_keeps_text_and_caption():
    from ouroboros.llm import LLMClient
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}, "_caption": "[browser shot]"},
        ],
    }]
    out = LLMClient._replace_image_blocks_with_placeholder(msgs, "local llama.cpp")
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "look at this"}
    assert blocks[1]["type"] == "text"
    assert "our local llama.cpp transport lane cannot carry images" in blocks[1]["text"]
    assert "[browser shot]" in blocks[1]["text"] and "model has no vision" not in blocks[1]["text"]
    # canonical transcript untouched (deep copy)
    assert msgs[0]["content"][1]["type"] == "image_url"
    # Text-only messages pass through untouched.
    text_only = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert LLMClient._replace_image_blocks_with_placeholder(text_only, "GigaChat") is text_only
