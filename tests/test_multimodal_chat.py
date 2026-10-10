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
    def test_names_are_not_evidence(self, tmp_path, monkeypatch):
        """Without a route's own catalog statement every API route is unknown, the
        names a former prefix table listed included; our local lane's limit is the
        send policy's transport fact, not a model verdict."""
        from ouroboros.provider_models import supports_vision

        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
        for model in ("openai/gpt-5.5", "google/gemini-3.5-flash", "anthropic::claude-opus-5",
                      "deepseek/deepseek-chat", "openai::o3-mini", "", "some-model (local)"):
            assert supports_vision(model) is None, model

    def test_a_recorded_catalog_statement_answers_only_for_its_route(self, tmp_path, monkeypatch):
        from ouroboros.provider_models import supports_vision
        from ouroboros.vision_routing import record_catalog_image_input

        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
        record_catalog_image_input("openrouter", "https://openrouter.ai/api/v1", [
            {"id": "deepseek/deepseek-vl", "architecture": {"input_modalities": ["text", "image"]}},
            {"id": "openai/gpt-5.5", "architecture": {"input_modalities": ["text"]}},
        ], source="OpenRouter /models")
        assert supports_vision("deepseek/deepseek-vl") is True
        assert supports_vision("openai/gpt-5.5") is False
        # The same slug on the direct OpenAI route is another route: still unknown.
        assert supports_vision("openai::gpt-5.5") is None


class TestWebAttachmentBlocks:
    def test_attachment_acceptance_binds_bytes_without_a_redundant_inline_image(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        import ouroboros.gateway.ws as ws_mod
        from ouroboros.chat_uploads import store_upload
        from tests.test_live_image_delivery import pixels

        stored, reference = store_upload(pixels(), "cat.png", data_dir=tmp_path)
        monkeypatch.setattr(ws_mod, "DATA_DIR", tmp_path)
        calls = []
        bridge = SimpleNamespace(ui_send=lambda text, **kwargs: calls.append((text, kwargs)))
        ws_mod._accept_with_attachments(bridge, "look", {"task_metadata": {}}, [
            {"filename": stored.name, "mime": "video/mp4", "display_name": "cat.png"},
            {"filename": "../settings.json", "mime": "image/png"},
        ])
        _text, kwargs = calls[0]
        assert "image_base64" not in kwargs
        metadata = kwargs["task_metadata"]
        assert metadata["chat_attachments"][0]["mime"] == "image/png"
        assert metadata["chat_attachment_uploads"][0]["sha256"] == reference["sha256"]
        assert metadata["chat_attachment_uploads"][1]["path"] == ""

    def test_build_user_content_attaches_caption_metadata(self):
        from ouroboros.context import build_user_content

        from tests.test_live_image_delivery import pixels
        content = build_user_content({
            "text": "look",
            "image_base64": base64.b64encode(pixels()).decode(),
            "image_mime": "image/png",
            "image_caption": "[user attachment: cat.png]",
        })
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert image_blocks and image_blocks[0]["_caption"] == "[user attachment: cat.png]"


class TestImageEviction:
    def test_keeps_only_newest_k(self):
        from ouroboros.loop_messages import _append_or_merge_user_content

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
        from ouroboros.loop_messages import _evict_stale_image_blocks

        block = _image_block("a", caption="screen")
        block["_source_path"] = "/data/uploads/screenshots/x.png"
        messages = [{"role": "user", "content": [block]}]
        _evict_stale_image_blocks(messages, incoming=MAX_LIVE_IMAGE_BLOCKS)
        text = messages[0]["content"][0]["text"]
        # Re-view hint points at view_image (local-file, native context, NOT web-gated);
        # VLM tools are also outside _WEB_TOOLS as of v6.45.
        assert "view_image path=/data/uploads/screenshots/x.png" in text


class TestImageTokenEstimates:
    def test_loop_estimate_uses_fixed_equivalent(self):
        from ouroboros.context_fit import estimate_context_prompt_tokens

        def messages(size):
            return [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{'A' * size}",
                    }},
            ],
            }]

        assert estimate_context_prompt_tokens(messages(10_000)) == (
            estimate_context_prompt_tokens(messages(1_000_000))
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
        from ouroboros.context_compaction import _atomic_units

        messages = [{
            "role": "assistant",
            "content": "inspect",
            "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "view", "arguments": "{}"},
            }],
        }, {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [_image_block("a", caption="login page")],
        }]
        units = _atomic_units(messages)
        assert len(units) == 1
        assert "login page" in units[0].source_text
        assert "base64" not in units[0].source_text

        messages[1]["content"] = [_image_block("a")]
        assert _atomic_units(messages) == ()

    def test_gigachat_text_placeholder(self):
        from ouroboros.llm import LLMClient

        text = LLMClient._gigachat_text([
            {"type": "text", "text": "hello "},
            _image_block("a"),
        ])
        assert "hello" in text
        # The marker names our lane, never the model.
        assert "[image omitted: our GigaChat transport lane cannot carry images" in text
        assert "model has no vision" not in text

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

    def test_vision_preparation_runs_outside_main_physical_binding(self, tmp_path, monkeypatch):
        from ouroboros import usage_accounting as ua
        from ouroboros.loop_llm_call import call_llm_with_retry
        from ouroboros.vision_routing import prepare_messages_for_send as real_prepare

        seen = {}

        def prepare(messages, *, routing):
            seen["prepare_context"] = ua.current_physical_attempt_context()
            seen["prepare_predicate"] = ua.current_physical_attempt_predicate()
            return real_prepare(messages, routing=routing)

        class LLM:
            def chat(self, **kwargs):
                seen["chat_context"] = ua.current_physical_attempt_context()
                seen["chat_predicate"] = ua.current_physical_attempt_predicate()
                return {"content": "ok"}, {}

        physical = ua.PhysicalAttemptContext(
            profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate",
            route_fp="route", round_id="round", target_total_tokens=None,
            capacity_total_tokens=500_000, context_target_miss=False,
            automatic_pass_used=False,
        )
        predicate = lambda request: True
        monkeypatch.setattr("ouroboros.vision_routing.prepare_messages_for_send", prepare)
        call_llm_with_retry(
            LLM(), [{"role": "user", "content": "plain"}], "openai/gpt-5.5",
            None, "medium", 1, tmp_path, "task", 1, None, {}, "task", False,
            physical_context=physical, candidate_predicate=predicate,
        )
        assert seen["prepare_context"] is None
        assert seen["prepare_predicate"] is None
        assert seen["chat_context"] == physical
        assert seen["chat_predicate"] is predicate


class TestNativeScreenshotInjection:
    def _inject(self, tmp_path, monkeypatch, model):
        import ouroboros.tools.browser as browser_mod

        monkeypatch.setenv("OUROBOROS_MODEL", model)

        class Ctx:
            drive_root = tmp_path
            messages = [{"role": "user", "content": "start"}]

        note = browser_mod._inject_native_screenshot(Ctx(), base64.b64encode(b"png").decode())
        return Ctx, note

    def test_injects_into_canonical_context_whatever_the_model(self, tmp_path, monkeypatch):
        """The screenshot is canonical input for every route; the send policy decides
        what a route receives, so the attach step never judges the model."""
        for index, model in enumerate(("openai/gpt-5.5", "deepseek/deepseek-chat")):
            ctx, note = self._inject(tmp_path / str(index), monkeypatch, model)
            # Stored in context is not the same as seen by the model: the note says so.
            assert "in your context as an image" in note and "natively" not in note
            content = ctx.messages[-1]["content"]
            assert isinstance(content, list)
            assert any(b.get("type") == "image_url" for b in content if isinstance(b, dict))
            shots = list((tmp_path / str(index) / "uploads" / "screenshots").glob("*.png"))
            assert shots, "screenshot must be persisted for re-view"
