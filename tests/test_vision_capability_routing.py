"""v6.37.0 guards (C2.1/C2.2/C2.3): the VLM lane must consult vision capability
before sending an image — route to a vision-capable slot or surface a typed
capability gap, never blind-send to a model that 404s and gets banged in a loop."""

from types import SimpleNamespace


def test_resolve_vlm_model_honors_vision_capability(monkeypatch):
    from ouroboros.tools import vision as V
    client = object()
    # explicit model: honored ONLY if it actually supports vision
    assert V._resolve_vlm_model(client, "google/gemini-3.5-flash") == "google/gemini-3.5-flash"
    assert V._resolve_vlm_model(client, "z-ai/glm-5.2") == ""  # explicit blind -> typed gap

    # no explicit model: first VISION-capable candidate wins (blind active skipped)
    monkeypatch.setattr(
        V, "_vision_capable_slot_candidates",
        lambda c, ctx=None: ["z-ai/glm-5.2", "google/gemini-3.5-flash", "openai/gpt-5.5"],
    )
    assert V._resolve_vlm_model(client, "", ctx=SimpleNamespace()) == "google/gemini-3.5-flash"

    # nothing vision-capable -> "" so the caller surfaces VLM_NO_VISION_MODEL
    monkeypatch.setattr(V, "_vision_capable_slot_candidates", lambda c, ctx=None: ["z-ai/glm-5.2"])
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


def test_analyze_screenshot_no_vision_returns_typed_gap(monkeypatch):
    from ouroboros.tools import vision as V
    monkeypatch.setattr(V, "_resolve_vlm_model", lambda *a, **k: "")
    ctx = SimpleNamespace(browser_state=SimpleNamespace(last_screenshot_b64="aGk="))
    out = V._analyze_screenshot(ctx, prompt="check")
    assert out == V._VLM_NO_VISION_MODEL_MSG
    assert "Do NOT retry the image" in out


def test_build_remote_kwargs_vision_gate_uses_qualified_identity(monkeypatch):
    """E1: the blind-model gate must consult supports_vision() with the QUALIFIED
    identity (usage_model) — the stripped resolved_model has lost its provider
    namespace, matched no vision prefix, and replaced the image with the omission
    placeholder on every direct-provider install (incl. the shipped default
    openai::gpt-5.6-terra). Same identity contract as the browser-screenshot
    call site (tools/browser.py)."""
    from ouroboros import provider_models
    from ouroboros.llm import LLMClient

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Static map authority only: a parallel test's overlay write must not flip verdicts.
    monkeypatch.setattr(provider_models, "_VISION_OVERLAY", {})
    client = LLMClient()
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }]

    # Qualified direct-provider identity keeps its image block (was blinded).
    direct = client._resolve_remote_target("openai::gpt-5.6-terra")
    kwargs = client._build_remote_kwargs(direct, msgs, "high", 128, "auto", None, None)
    assert [b.get("type") for b in kwargs["messages"][0]["content"]] == ["text", "image_url"]

    # OpenRouter identities behave exactly as before: vision-capable keeps the
    # image, a blind model still gets the honest omission placeholder.
    sol = client._resolve_remote_target("openai/gpt-5.6-sol")
    kwargs = client._build_remote_kwargs(
        sol, msgs, "high", 128, "auto", None, None, skip_capability_fetch=True
    )
    assert [b.get("type") for b in kwargs["messages"][0]["content"]] == ["text", "image_url"]

    blind = client._resolve_remote_target("deepseek/deepseek-chat")
    kwargs = client._build_remote_kwargs(
        blind, msgs, "high", 128, "auto", None, None, skip_capability_fetch=True
    )
    blocks = kwargs["messages"][0]["content"]
    assert [b.get("type") for b in blocks] == ["text", "text"]
    assert "image omitted" in blocks[1]["text"]


def test_replace_image_blocks_with_placeholder_keeps_text_and_caption():
    from ouroboros.llm import LLMClient
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}, "_caption": "[browser shot]"},
        ],
    }]
    out = LLMClient._replace_image_blocks_with_placeholder(msgs)
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "look at this"}
    assert blocks[1]["type"] == "text"
    assert "image omitted" in blocks[1]["text"] and "[browser shot]" in blocks[1]["text"]
    # canonical transcript untouched (deep copy)
    assert msgs[0]["content"][1]["type"] == "image_url"
