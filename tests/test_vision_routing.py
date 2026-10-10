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


def test_inline_mode_sends_pixels_even_when_metadata_says_no(monkeypatch):
    """Owner decision: Inline sends the image even against a "no"; the provider's own
    refusal is what the agent sees, never a pre-emptive placeholder."""
    from ouroboros import vision_routing
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    for verdict in (None, False):
        monkeypatch.setattr(vision_routing, "supports_vision", lambda *_a, verdict=verdict, **_k: verdict)
        messages = _image_message()
        assert prepare_messages_for_send(
            messages, routing=VisionRoutingContext("not/vision", object(), {}),
        ) is messages
    # The same "no" does withhold pixels in Auto (here: the block's own caption).
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setattr(vision_routing, "supports_vision", lambda *_a, **_k: False)
    out = prepare_messages_for_send(_image_message(), routing=VisionRoutingContext("not/vision", object(), {}))
    assert out[0]["content"][1] == {"type": "text", "text": "[image caption: old caption]"}


def test_an_unreadable_route_fact_is_unknown_not_no(monkeypatch):
    """Auto withholds pixels only on a fact; a failure to read one sends as unknown,
    while a model wait still belongs to its owner."""
    import pytest

    from ouroboros import vision_routing
    from ouroboros.model_wait import ModelWaitInterrupted
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")

    def unreadable(*_a, **_k):
        raise RuntimeError("catalog read failed")

    monkeypatch.setattr(vision_routing, "supports_vision", unreadable)
    messages = _image_message()
    assert prepare_messages_for_send(messages, routing=VisionRoutingContext("acme/x", object(), {})) is messages

    def interrupted(*_a, **_k):
        raise ModelWaitInterrupted("owner stop", role="main")

    monkeypatch.setattr(vision_routing, "supports_vision", interrupted)
    with pytest.raises(ModelWaitInterrupted):
        prepare_messages_for_send(_image_message(), routing=VisionRoutingContext("acme/x", object(), {}))


def test_inline_mode_on_our_local_lane_names_the_lane(monkeypatch):
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    out = prepare_messages_for_send(
        _image_message(), routing=VisionRoutingContext("some-model (local)", object(), {}, use_local=True),
    )
    assert out[0]["content"][1]["text"].startswith("[image omitted: our local llama.cpp transport lane")


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
    import time
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FakeLLM:
        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *args, **kwargs):
            self.timeout = kwargs["timeout"]
            return "fresh caption", {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.01}

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "0")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")

    events = queue.Queue()
    llm = FakeLLM()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext(
            "not/vision", llm, {}, drive_root=tmp_path, task_id="task-1",
            event_queue=events, deadline_ts=time.time() + 10,
        ),
    )

    assert out[0]["content"][1]["text"] == "[image caption: fresh caption]"
    assert 0 < llm.timeout <= 10
    calls = list((tmp_path / "observability" / "calls").rglob("*.json"))
    assert calls
    event_rows = []
    while not events.empty():
        event_rows.append(events.get_nowait())
    assert any(event.get("source") == "vision_caption" for event in event_rows)
    operation_rows = [event for event in event_rows if event.get("type") == "cognitive_operation"]
    assert [event.get("phase") for event in operation_rows] == ["started", "finished"]


def test_expired_caption_window_does_not_dispatch_model(monkeypatch):
    import time
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FakeLLM:
        def vision_query(self, *args, **kwargs):
            raise AssertionError("expired owner window must not dispatch a caption")

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext(
            "not/vision", FakeLLM(), {}, deadline_ts=time.time() - 10,
        ),
    )

    assert "caption unavailable" in out[0]["content"][1]["text"]


def test_caption_does_not_start_inside_finalization_reserve(monkeypatch):
    import time
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FakeLLM:
        def vision_query(self, *args, **kwargs):
            raise AssertionError("reserve-only owner window must not dispatch a caption")

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext(
            "not/vision", FakeLLM(), {}, deadline_ts=time.time() + 5,
        ),
    )

    assert "caption unavailable" in out[0]["content"][1]["text"]


def test_caption_persistence_failure_keeps_paid_caption_one_terminal(monkeypatch, tmp_path):
    """capinv-447 D9. This test previously PINNED the defect: it asserted the
    persistence failure replaced the paid caption with "caption unavailable".
    The real contract is the terminal-event sequence (started + exactly one
    terminal); the model call SUCCEEDED, so the caller must receive the fresh
    caption and the operation must terminate "finished", not "failed"."""
    import queue
    from ouroboros import vision_routing
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FakeLLM:
        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *args, **kwargs):
            return "fresh caption", {"cost": 0.01}

    def fake_persist(_root, **kwargs):
        if kwargs["call_type"] == "vision_caption_response":
            raise OSError("response receipt unavailable")
        return {"manifest_ref": "request.json"}

    monkeypatch.setattr(vision_routing, "persist_call", fake_persist)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    messages = _image_message()
    messages[0]["content"][1].pop("_caption")
    events = queue.Queue()
    out = prepare_messages_for_send(
        messages,
        routing=VisionRoutingContext(
            "not/vision", FakeLLM(), {}, drive_root=tmp_path,
            task_id="task-1", event_queue=events,
        ),
    )
    assert out[0]["content"][1]["text"] == "[image caption: fresh caption]"
    phases = []
    while not events.empty():
        event = events.get_nowait()
        if event.get("type") == "cognitive_operation":
            phases.append(event.get("phase"))
    assert phases == ["started", "finished"]


def test_failed_caption_is_not_memoized_second_attempt_retries(monkeypatch):
    """capinv-447 D9: a failure label must never be memoized — memoizing it used
    to block any second caption attempt for that image for the rest of the task."""
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    class FlakyLLM:
        calls = 0

        def default_model(self):
            return "google/gemini-3.5-flash"

        def vision_query(self, *args, **kwargs):
            FlakyLLM.calls += 1
            if FlakyLLM.calls == 1:
                raise RuntimeError("transient provider failure")
            return "second try caption", {"cost": 0.01}

    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    monkeypatch.setenv("OUROBOROS_MODEL_VISION", "google/gemini-3.5-flash")
    llm = FlakyLLM()
    shared_usage: dict = {}  # one accumulated_usage dict = one per-task memo
    first = _image_message()
    first[0]["content"][1].pop("_caption")
    out1 = prepare_messages_for_send(
        first, routing=VisionRoutingContext("not/vision", llm, shared_usage),
    )
    assert "caption unavailable" in out1[0]["content"][1]["text"]

    second = _image_message()
    second[0]["content"][1].pop("_caption")
    out2 = prepare_messages_for_send(
        second, routing=VisionRoutingContext("not/vision", llm, shared_usage),
    )
    assert out2[0]["content"][1]["text"] == "[image caption: second try caption]"
    assert FlakyLLM.calls == 2

    # The SUCCESSFUL caption is still memoized: a third send pays nothing.
    third = _image_message()
    third[0]["content"][1].pop("_caption")
    out3 = prepare_messages_for_send(
        third, routing=VisionRoutingContext("not/vision", llm, shared_usage),
    )
    assert out3[0]["content"][1]["text"] == "[image caption: second try caption]"
    assert FlakyLLM.calls == 2


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


def test_vlm_tools_are_not_web_resource_gated():
    from ouroboros.tools.registry import _WEB_TOOLS

    assert "web_search" in _WEB_TOOLS
    assert "browse_page" in _WEB_TOOLS
    assert "browser_action" in _WEB_TOOLS
    assert "vlm_query" not in _WEB_TOOLS
    assert "analyze_screenshot" not in _WEB_TOOLS
