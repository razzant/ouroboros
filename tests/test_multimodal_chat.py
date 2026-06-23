"""Native multimodal chat (WS-H, v6.26.0): vision capability, attachment
blocks, eviction, image-aware token estimates, compaction safety, and
non-vision lane placeholders."""

import base64
import json

from ouroboros.context_budget import IMAGE_BLOCK_CHAR_EQUIVALENT, MAX_LIVE_IMAGE_BLOCKS


def _image_block(tag: str = "x", caption: str = "") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{tag * 8}"},
        "_caption": caption,
    }


class TestSupportsVision:
    def test_static_map(self):
        from ouroboros.provider_models import supports_vision

        assert supports_vision("openai/gpt-5.5") is True
        assert supports_vision("google/gemini-3.5-flash") is True
        assert supports_vision("anthropic::claude-opus-4-8") is True
        assert supports_vision("deepseek/deepseek-chat") is False
        assert supports_vision("") is False
        assert supports_vision("some-model (local)") is False

    def test_overlay_wins(self):
        from ouroboros import provider_models

        provider_models.update_vision_overlay("deepseek/deepseek-vl", True)
        assert provider_models.supports_vision("deepseek/deepseek-vl") is True
        provider_models.update_vision_overlay("openai/gpt-5.5", False)
        try:
            assert provider_models.supports_vision("openai/gpt-5.5") is False
        finally:
            provider_models._VISION_OVERLAY.pop("openai/gpt-5.5", None)
            provider_models._VISION_OVERLAY.pop("deepseek/deepseek-vl", None)


class TestWebAttachmentBlocks:
    def test_first_image_attachment_resolves_upload(self, tmp_path, monkeypatch):
        import ouroboros.gateway.ws as ws_mod

        uploads = tmp_path / "uploads"
        uploads.mkdir(parents=True)
        (uploads / "abc_cat.png").write_bytes(b"\x89PNG fake")
        monkeypatch.setattr(ws_mod, "DATA_DIR", tmp_path)

        b64, mime, caption = ws_mod._first_image_attachment([
            {"filename": "abc_cat.png", "mime": "image/png", "display_name": "cat.png"},
        ])
        assert base64.b64decode(b64) == b"\x89PNG fake"
        assert mime == "image/png"
        assert "cat.png" in caption

    def test_traversal_and_non_image_rejected(self, tmp_path, monkeypatch):
        import ouroboros.gateway.ws as ws_mod

        (tmp_path / "uploads").mkdir(parents=True)
        secret = tmp_path / "settings.json"
        secret.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(ws_mod, "DATA_DIR", tmp_path)

        assert ws_mod._first_image_attachment(
            [{"filename": "../settings.json", "mime": "image/png"}]
        ) == ("", "", "")
        assert ws_mod._first_image_attachment(
            [{"filename": "doc.pdf", "mime": "application/pdf"}]
        ) == ("", "", "")

    def test_build_user_content_attaches_caption_metadata(self):
        from ouroboros.context import build_user_content

        content = build_user_content({
            "text": "look",
            "image_base64": "QUJD",
            "image_mime": "image/png",
            "image_caption": "[user attachment: cat.png]",
        })
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert image_blocks and image_blocks[0]["_caption"] == "[user attachment: cat.png]"


class TestImageEviction:
    def test_keeps_only_newest_k(self):
        from ouroboros.loop import _append_or_merge_user_content

        messages = []
        for idx in range(MAX_LIVE_IMAGE_BLOCKS + 2):
            _append_or_merge_user_content(
                messages,
                [
                    {"type": "text", "text": f"img {idx}"},
                    _image_block(str(idx), caption=f"shot-{idx}"),
                ],
            )
            messages.append({"role": "assistant", "content": "ok"})

        live = [
            block
            for msg in messages
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == "image_url"
        ]
        assert len(live) == MAX_LIVE_IMAGE_BLOCKS
        rendered = json.dumps(messages, ensure_ascii=False)
        assert "[image evicted: shot-0]" in rendered
        assert "[image evicted: shot-1]" in rendered

    def test_placeholder_includes_reviewable_path(self):
        from ouroboros.loop import _evict_stale_image_blocks

        block = _image_block("a", caption="screen")
        block["_source_path"] = "/data/uploads/screenshots/x.png"
        messages = [{"role": "user", "content": [block]}]
        _evict_stale_image_blocks(messages, incoming=MAX_LIVE_IMAGE_BLOCKS)
        text = messages[0]["content"][0]["text"]
        # Re-view hint points at view_image (local-file, native context, NOT web-gated);
        # vlm_query is in _WEB_TOOLS and is blocked under allowed_resources.web=false.
        assert "view_image path=/data/uploads/screenshots/x.png" in text


class TestImageTokenEstimates:
    def test_loop_estimate_uses_fixed_equivalent(self):
        from ouroboros.loop import _estimate_messages_chars

        huge_b64 = "A" * 1_000_000
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge_b64}"}},
            ],
        }]
        total = _estimate_messages_chars(messages)
        assert total < IMAGE_BLOCK_CHAR_EQUIVALENT + 1000, (
            "image block must count as the fixed equivalent, not base64 length"
        )

    def test_llm_estimate_symmetric(self):
        from ouroboros.llm import _estimate_message_chars

        huge_b64 = "A" * 1_000_000
        messages = [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": huge_b64}}],
        }]
        assert _estimate_message_chars(messages) == IMAGE_BLOCK_CHAR_EQUIVALENT


class TestCompactionAndLanes:
    def test_render_round_block_replaces_image(self):
        from ouroboros.context_compaction import _render_round_block

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "see this"},
                _image_block("a", caption="login page"),
            ],
        }]
        rendered = _render_round_block(messages, 0, 0)
        assert "[image: login page]" in rendered
        assert "base64" not in rendered

    def test_gigachat_text_placeholder(self):
        from ouroboros.llm import LLMClient

        text = LLMClient._gigachat_text([
            {"type": "text", "text": "hello "},
            _image_block("a"),
        ])
        assert "hello" in text
        assert "[image omitted: model has no vision" in text

    def test_provider_payload_strips_internal_metadata(self):
        from ouroboros.llm import LLMClient

        cleaned = LLMClient._copy_messages_with_cache_policy(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "u"}, "_caption": "c", "_source_path": "p"},
            ]}],
            allow_message_cache_control=False,
            flatten_tool_content_blocks=False,
        )
        block = cleaned[0]["content"][0]
        assert "_caption" not in block and "_source_path" not in block


class TestNativeScreenshotInjection:
    def test_injects_for_vision_model(self, tmp_path, monkeypatch):
        import ouroboros.tools.browser as browser_mod

        monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.5")

        class Ctx:
            drive_root = tmp_path
            messages = [{"role": "user", "content": "start"}]

        note = browser_mod._inject_native_screenshot(Ctx(), base64.b64encode(b"png").decode())
        assert "natively" in note
        content = Ctx.messages[-1]["content"]
        assert isinstance(content, list)
        assert any(b.get("type") == "image_url" for b in content if isinstance(b, dict))
        shots = list((tmp_path / "uploads" / "screenshots").glob("*.png"))
        assert shots, "screenshot must be persisted for re-view"

    def test_skipped_for_non_vision_model(self, tmp_path, monkeypatch):
        import ouroboros.tools.browser as browser_mod

        monkeypatch.setenv("OUROBOROS_MODEL", "deepseek/deepseek-chat")

        class Ctx:
            drive_root = tmp_path
            messages = [{"role": "user", "content": "start"}]

        note = browser_mod._inject_native_screenshot(Ctx(), base64.b64encode(b"png").decode())
        assert note == ""
        assert Ctx.messages[-1]["content"] == "start"
