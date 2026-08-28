from __future__ import annotations


def _image_message():
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aaa"}, "_caption": "old caption"},
        ],
    }]


def test_vision_config_slot_and_legacy_migration(monkeypatch):
    from ouroboros import config

    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.5")
    monkeypatch.delenv("OUROBOROS_MODEL_VISION", raising=False)
    assert config.get_vision_model() == "openai/gpt-5.5"
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-2.5-pro")
    assert config.get_vision_model() == "google/gemini-2.5-pro"

    settings = {"OUROBOROS_VISION_MODEL": "anthropic/claude-sonnet-4.5"}
    config.migrate_legacy_slot_keys(settings)
    assert settings["OUROBOROS_MODEL_VISION"] == "anthropic/claude-sonnet-4.5"
    assert "OUROBOROS_VISION_MODEL" not in settings


def test_auto_mode_keeps_inline_for_vision_model(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    messages = _image_message()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("google/gemini-3.5-flash", object(), {}),
    )
    assert out is messages


def test_auto_mode_treats_local_route_as_blind(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    messages = _image_message()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("google/gemini-3.5-flash", object(), {}, use_local=True),
    )
    assert out is not messages
    assert out[0]["content"][1]["text"] == "[image caption: old caption]"


def test_blind_route_text_only_transcript_avoids_copy(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    messages = [{"role": "user", "content": "hello"}]
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("not/vision", object(), {}),
    )
    assert out is messages


def test_caption_mode_rewrites_send_copy_without_mutating_transcript(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    messages = _image_message()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("not/vision", object(), {}),
    )

    assert out is not messages
    assert messages[0]["content"][1]["type"] == "image_url"
    assert out[0]["content"][1] == {"type": "text", "text": "[image caption: old caption]"}


def test_inline_mode_blind_model_fails_closed_without_caption(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    messages = _image_message()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("not/vision", object(), {}),
    )

    assert out is not messages
    assert out[0]["content"][1]["text"].startswith("[image omitted:")
    assert "old caption" not in out[0]["content"][1]["text"]


def test_off_mode_ignores_existing_caption(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "off")
    out = prepare_messages_for_send(
        _image_message(),
        routing=VisionRoutingContext("google/gemini-3.5-flash", object(), {}),
    )

    assert out[0]["content"][1]["text"].startswith("[image omitted:")
    assert "old caption" not in out[0]["content"][1]["text"]


def test_caption_call_records_observability(monkeypatch, tmp_path):
    import queue
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FakeLLM:
        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *args, **kwargs):
            return "fresh caption", {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01}

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")

    events = queue.Queue()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("not/vision", FakeLLM(), {}, drive_root=tmp_path, task_id="task-1", event_queue=events),
    )

    assert out[0]["content"][1]["text"] == "[image caption: fresh caption]"
    calls = list((tmp_path / "observability" / "calls").rglob("*.json"))
    assert calls
    assert events.get_nowait()["source"] == "vision_caption"


def test_caption_mode_does_not_treat_bracket_label_as_real_caption(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FakeLLM:
        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *args, **kwargs):
            return "actual visual caption", {"cost": 0}

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    messages = _image_message()
    messages[0]["content"][1]["_caption"] = "[image: file.png]"

    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext("not/vision", FakeLLM(), {}),
    )

    assert out[0]["content"][1]["text"] == "[image caption: actual visual caption]"


def _vision_provider_error(*, observed: bool, status: int | None = None):
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    exc = ConnectionError("caption request connection ended")
    exc.physical_attempt_capture = PhysicalAttemptCapture(
        attempt_id="caption-attempt",
        model="google/gemini-3.5-flash",
        provider="openrouter",
        state="unresolved",
        provider_response_observed=observed,
        provider_status_code=status,
    )
    return exc


def test_unknown_caption_outcome_blocks_main_model_send(monkeypatch, tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry

    calls = {"caption": 0, "main": 0}

    class FakeLLM:
        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *_args, **_kwargs):
            calls["caption"] += 1
            raise _vision_provider_error(observed=False)

        def chat(self, **_kwargs):
            calls["main"] += 1
            raise AssertionError("main send must not follow an unknown caption outcome")

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")
    usage = {}
    msg, _cost = call_llm_with_retry(
        FakeLLM(), messages, "not/vision", None, "medium", 3,
        tmp_path / "logs", "caption-unknown", 1, None, usage,
        attempt_cap=2,
    )

    assert msg is None
    assert calls == {"caption": 1, "main": 0}
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"


def test_response_observed_caption_failure_keeps_fail_soft_main_send(monkeypatch, tmp_path):
    from ouroboros.loop_llm_call import call_llm_with_retry

    calls = {"caption": 0, "main": 0}

    class FakeLLM:
        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *_args, **_kwargs):
            calls["caption"] += 1
            raise _vision_provider_error(observed=True, status=503)

        def chat(self, **_kwargs):
            calls["main"] += 1
            return {"content": "continued safely"}, {
                "provider": "openrouter", "resolved_model": "not/vision",
            }

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")
    msg, _cost = call_llm_with_retry(
        FakeLLM(), messages, "not/vision", None, "medium", 3,
        tmp_path / "logs", "caption-known", 1, None, {},
        attempt_cap=2,
    )

    assert msg["content"] == "continued safely"
    assert calls == {"caption": 1, "main": 1}


def test_vlm_tools_are_not_web_resource_gated():
    from ouroboros.tools.registry import _WEB_TOOLS

    assert "web_search" in _WEB_TOOLS
    assert "browse_page" in _WEB_TOOLS
    assert "browser_action" in _WEB_TOOLS
    assert "vlm_query" not in _WEB_TOOLS
    assert "analyze_screenshot" not in _WEB_TOOLS
