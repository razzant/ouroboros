"""Native multimodal chat (WS-H, v6.26.0): vision capability, attachment
blocks, eviction, image-aware token estimates, compaction safety, and
non-vision lane placeholders."""

import base64
from copy import deepcopy
import json

import pytest

from ouroboros.context_budget import IMAGE_BLOCK_CHAR_EQUIVALENT, MAX_LIVE_IMAGE_BLOCKS, LIVE_IMAGE_BLOCKS_HIGH_WATER


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
    @pytest.mark.parametrize("count", range(6, LIVE_IMAGE_BLOCKS_HIGH_WATER + 1))
    def test_up_to_high_water_appends_without_changing_sent_images(self, count):
        from ouroboros.loop_messages import _append_or_merge_user_content

        messages = [{"role": "user", "content": [_image_block(str(idx))]} for idx in range(count - 1)]
        sent = deepcopy(messages)
        _append_or_merge_user_content(messages, [_image_block(str(count - 1))])
        assert messages[:-1] == sent
        assert sum(block.get("type") == "image_url" for msg in messages for block in msg["content"]) == count

    def test_eleventh_image_retires_oldest_six_together(self):
        from ouroboros.loop_messages import _append_or_merge_user_content

        messages = []
        for idx in range(LIVE_IMAGE_BLOCKS_HIGH_WATER + 1):
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
        assert [block["_caption"] for block in live] == [f"shot-{idx}" for idx in range(6, 11)]
        rendered = json.dumps(messages, ensure_ascii=False)
        assert all(f"[image evicted: shot-{idx}]" in rendered for idx in range(6))

    @pytest.mark.parametrize(("prior", "incoming"), [(5, 6), (9, 3), (0, 11), (2, 14)])
    def test_incoming_batch_keeps_five_newest_even_when_batch_exceeds_high_water(self, prior, incoming):
        from ouroboros.loop_messages import _append_or_merge_user_content

        messages = [{"role": "user", "content": [_image_block(str(idx), caption=f"shot-{idx}")]}
                    for idx in range(prior)]
        _append_or_merge_user_content(messages, [
            _image_block(str(idx), caption=f"shot-{idx}") for idx in range(prior, prior + incoming)])
        blocks = [block for row in messages for block in row["content"]]
        live = [block["_caption"] for block in blocks if block.get("type") == "image_url"]
        assert live == [f"shot-{idx}" for idx in range(prior + incoming - MAX_LIVE_IMAGE_BLOCKS, prior + incoming)]
        assert sum("[image evicted:" in block.get("text", "") for block in blocks) == prior + incoming - len(live)

    def test_eviction_names_its_change_of_sent_history_and_a_fit_names_none(self):
        """The evicted bytes were already sent: the next observation says who changed them."""
        from types import SimpleNamespace

        from ouroboros.loop_messages import _append_or_merge_user_content
        from ouroboros.transcript_prefix import SANCTION_ATTR

        for prior, origin in ((LIVE_IMAGE_BLOCKS_HIGH_WATER, "image_eviction"), (1, None)):
            slot = SimpleNamespace()
            messages = [{"role": "user", "content": [_image_block(str(idx))]} for idx in range(prior)]
            _append_or_merge_user_content(messages, [_image_block("new")], slot=slot)
            assert getattr(slot, SANCTION_ATTR, None) == origin

    def test_a_tool_image_names_its_eviction_on_the_live_slot(self, monkeypatch):
        """view_image and the host's auto-attach run on a staged copy; the stamp belongs to the live slot."""
        from types import SimpleNamespace

        from ouroboros.tools import vision
        from ouroboros.transcript_prefix import SANCTION_ATTR

        live = SimpleNamespace(messages=[{"role": "user", "content": [_image_block(str(idx))]}
                                         for idx in range(LIVE_IMAGE_BLOCKS_HIGH_WATER)])
        monkeypatch.setattr(vision, "_load_local_image_payload", lambda _ctx, _path, retain_original=False: (
            {"base64": "QUFB", "mime": "image/png", "source_path": "/tmp/shot.png", "source_sha256": "fixture"}, ""))
        staged = SimpleNamespace(messages=live.messages)
        ok, _note = vision.attach_local_image_to_context(staged, "shot.png", stamp=live)
        assert ok and getattr(live, SANCTION_ATTR, None) == "image_eviction"
        assert getattr(staged, SANCTION_ATTR, None) is None

    def test_placeholder_includes_reviewable_path(self):
        from ouroboros.loop_messages import _evict_stale_image_blocks

        block = _image_block("a", caption="screen")
        block["_source_path"] = "/data/uploads/screenshots/x.png"
        messages = [{"role": "user", "content": [block]}]
        _evict_stale_image_blocks(messages, incoming=LIVE_IMAGE_BLOCKS_HIGH_WATER)
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
            [{"role": "tool", "tool_call_id": "i", "content": "staged", "_review_image_id": "1:i"},
             {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "u"}, "_caption": "c", "_source_path": "p",
                 "_source_sha256": "hash", "_review_image_id": "1:i"},
            ]}],
            allow_message_cache_control=False,
            flatten_tool_content_blocks=False,
        )
        assert "_review_image_id" not in cleaned[0] and cleaned[0]["tool_call_id"] == "i"
        block = cleaned[1]["content"][0]
        assert "_caption" not in block and "_source_path" not in block
        assert "_source_sha256" not in block and "_review_image_id" not in block
        assert block["image_url"] == {"url": "u"}

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
