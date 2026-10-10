"""Subscription images reach the ordinary Main and registered VLM entrypoints."""

import base64
from copy import deepcopy
import json
import os
from types import SimpleNamespace

import pytest

from ouroboros import provider_models
from ouroboros import vision_routing
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.llm import LLMClient
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.subscription_install_presets import compile_model_settings
from ouroboros.tools import vision
from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send
from tests.test_llm_claudexor import MODEL, ledger, setup as _subscription_transport
from tests.test_vision_model_wait import child_fixture as _child_fixture, _events

subscription_transport = _subscription_transport
child_fixture = _child_fixture


@pytest.fixture
def catalog(monkeypatch):
    calls = []
    # image_input: True/False renders the engine's imageInput claim; None renders
    # the row WITHOUT the field — the adversarial shape, because inputModalities
    # still claim ["text", "image"] on every row while the engine claims no
    # image transport.
    state = {"image_input": True, "error": None}

    def read(source, profile=None, *, requested_model=None):
        calls.append((source, profile, requested_model))
        if state["error"]:
            raise ClaudexorUnavailable(state["error"], "controlled metadata unavailable")
        row = {"id": requested_model, "inputModalities": ["text", "image"]}
        if state["image_input"] is not None:
            row["imageInput"] = state["image_input"]
        return {"source": source, "credentialProfileId": profile or "auto-account",
                "accountFingerprint": "identity-" + (profile or "auto-account"),
                "models": [row]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(read))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", json.dumps({"main": "main-a", "vision": "vision-b"}))
    return state, calls


def image_messages():
    return [{"role": "system", "content": "Own SYSTEM and BIBLE 🐍\r\n"},
            {"role": "user", "content": [{"type": "text", "text": "Inspect this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]


def install_preset(monkeypatch, model=MODEL):
    proposed = compile_model_settings([{"value": model, "is_default": True,
                                        "input_modalities": ["text", "image"]}], {})
    for key, value in proposed.items():
        monkeypatch.setenv(key, value)
    assert proposed["OUROBOROS_MODEL"] == proposed["OUROBOROS_MODEL_VISION"] == model
    return proposed


@pytest.mark.parametrize("metadata_error", [None, "daemon_not_discovered", "subscription_window_exhausted"])
def test_preset_images_reach_real_main_transport(subscription_transport, catalog, monkeypatch, metadata_error):
    from ouroboros.loop_llm_call import call_llm_with_retry

    root, gateway, client = subscription_transport
    state, calls = catalog
    state["error"] = metadata_error
    proposed = install_preset(monkeypatch)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    messages = image_messages()
    original = deepcopy(messages)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    answer, _ = call_llm_with_retry(client, messages, proposed["OUROBOROS_MODEL"], [],
        "high", 1, root / "logs", "task-one", 1, None, {},
        model_role="main", model_account_override="explicit-main")
    assert answer["content"] == "Ответ 🐍"
    assert gateway.uploads[0][0]["messages"] == original
    assert gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "explicit-main"}
    assert calls == [("codex", "explicit-main", "exact-model")]
    assert messages == original
    assert len(gateway.creates) == 1 and len(gateway.accepted_operations) == 1
    assert [(row["state"], row["revision"]) for row in ledger(root)] == [("settled", 3)]


def test_image_capability_is_exact_role_account_and_not_a_global_record(catalog, monkeypatch, tmp_path):
    from ouroboros import capability_evidence

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    state, calls = catalog
    assert provider_models.supports_vision(MODEL, model_role="main") is True
    state["image_input"] = False
    assert provider_models.supports_vision(MODEL, model_role="vision") is False
    state["image_input"] = True
    assert provider_models.supports_vision(MODEL, model_role="vision", model_account_override="") is True
    assert calls == [("codex", "main-a", "exact-model"), ("codex", "vision-b", "exact-model"),
                     ("codex", None, "exact-model")]
    # An account's catalog answer is read per call; it never becomes a stored route record.
    store = capability_evidence._load(capability_evidence.canonical_evidence_root())
    assert store["image_input"] == {}


def test_absent_image_input_field_is_an_honest_refusal_not_unknown(catalog, monkeypatch):
    """Adversarial invariant: modalities claim images, the engine does not.

    The catalog row still advertises inputModalities=["text","image"], but the
    entry carries NO imageInput field. Under the imageInput contract an
    accessible catalog row without the field is an engine that claims no image
    transport: supports_vision returns False (honest refusal), never unknown.
    Semantic change vs the old inputModalities reader: this shape used to stay
    unknown and keep the image inline; now auto mode rewrites the input
    (placeholder/caption) while the canonical messages keep the image block.
    """
    state, _ = catalog
    state["image_input"] = None  # renders the row WITHOUT the imageInput field
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    assert provider_models.supports_vision(MODEL, model_role="main") is False
    # No caption model resolvable: the rewrite lands on the omission placeholder.
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *a, **kw: "")
    messages = image_messages()
    prepared = prepare_messages_for_send(messages, routing=VisionRoutingContext(MODEL, object(), {}))
    assert prepared is not messages
    assert "image omitted" in prepared[1]["content"][1]["text"]
    assert messages[1]["content"][1]["type"] == "image_url"  # canonical input is never erased


def test_absent_image_input_field_places_the_placeholder_in_the_real_payload(
        subscription_transport, catalog, monkeypatch):
    """The same honest refusal, read at the gateway payload: no image block leaves."""
    from ouroboros.loop_llm_call import call_llm_with_retry

    root, gateway, client = subscription_transport
    state, _ = catalog
    state["image_input"] = None
    install_preset(monkeypatch)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *a, **kw: "")
    (root / "logs").mkdir(parents=True, exist_ok=True)
    answer, _ = call_llm_with_retry(client, image_messages(), MODEL, [], "high", 1,
        root / "logs", "task-one", 1, None, {}, model_role="main")
    assert answer["content"] == "Ответ 🐍"
    sent = gateway.uploads[0][0]["messages"]
    assert "image omitted" in sent[1]["content"][1]["text"]
    assert not any(b.get("type") == "image_url" for m in sent
                   if isinstance(m.get("content"), list) for b in m["content"])


@pytest.mark.parametrize("error", ["daemon_not_discovered", "subscription_window_exhausted"])
def test_unavailable_catalog_keeps_the_unknown_inline_contract(catalog, monkeypatch, error):
    """ClaudexorUnavailable → None: input preserved, the real call can refuse typed."""
    state, _ = catalog
    state["error"] = error
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    assert provider_models.supports_vision(MODEL, model_role="main") is None
    messages = image_messages()
    assert prepare_messages_for_send(messages, routing=VisionRoutingContext(MODEL, object(), {})) is messages


def test_image_input_true_sends_inline_blocks_through_the_real_transport(subscription_transport, catalog, monkeypatch):
    """imageInput=true → supports_vision True → inline blocks reach the gateway payload."""
    from ouroboros.loop_llm_call import call_llm_with_retry

    root, gateway, client = subscription_transport
    state, calls = catalog
    state["image_input"] = True
    install_preset(monkeypatch)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    messages = image_messages()
    original = deepcopy(messages)
    assert prepare_messages_for_send(messages, routing=VisionRoutingContext(MODEL, client, {})) is messages
    (root / "logs").mkdir(parents=True, exist_ok=True)
    answer, _ = call_llm_with_retry(client, deepcopy(messages), MODEL, [], "high", 1,
        root / "logs", "task-one", 1, None, {}, model_role="main")
    assert answer["content"] == "Ответ 🐍"
    sent = gateway.uploads[0][0]["messages"]
    block = next(b for m in sent if isinstance(m.get("content"), list)
                 for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url")
    assert block["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert "_caption" not in block and "_source_path" not in block
    assert messages == original
    # Two catalog reads: the explicit routing gate above, then send-time
    # preparation inside call_llm_with_retry — both on the exact main account.
    assert calls == [("codex", "main-a", "exact-model"), ("codex", "main-a", "exact-model")]


def test_metadata_cannot_borrow_another_account_or_source(monkeypatch):
    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(lambda *a, **k: {
        "source": "codex", "credentialProfileId": "another-account",
        "models": [{"id": "exact-model", "inputModalities": ["text"]}]}))
    assert provider_models.supports_vision(MODEL, model_account_override="wanted") is None
    assert provider_models.supports_vision("claudexor::another=exact-model") is None


def test_text_only_and_image_off_do_not_query_catalog(catalog, monkeypatch):
    _, calls = catalog
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    text = [{"role": "user", "content": "hello"}]
    assert prepare_messages_for_send(text, routing=VisionRoutingContext(MODEL, object(), {})) is text
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "off")
    prepared = prepare_messages_for_send(image_messages(), routing=VisionRoutingContext(MODEL, object(), {}))
    assert "image omitted" in prepared[1]["content"][1]["text"]
    assert calls == []


def test_confirmed_nonvision_is_still_called_when_named_and_sent_inline(catalog, monkeypatch):
    """Owner decision (Inline = always send): a model named explicitly, and an Inline
    send, carry the image even though this account's catalog says text only; the
    route's own refusal is what the agent then sees. Auto withholds on that "no"."""
    state, calls = catalog
    state["image_input"] = False
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    assert vision._resolve_vlm_model(LLMClient(), MODEL) == MODEL
    messages = image_messages()
    assert prepare_messages_for_send(messages, routing=VisionRoutingContext(MODEL, object(), {})) is messages
    assert calls == [], "neither an explicit model nor Inline asks the catalog for permission"
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    for key, value in (("OUROBOROS_MODEL", MODEL), ("OUROBOROS_MODEL_VISION", ""),
                       ("OUROBOROS_MODEL_LIGHT", ""), ("OUROBOROS_MODEL_FALLBACKS", "")):
        monkeypatch.setenv(key, value)  # every caption candidate is this text-only route
    prepared = prepare_messages_for_send(image_messages(), routing=VisionRoutingContext(MODEL, object(), {}))
    assert "image omitted" in prepared[1]["content"][1]["text"]
    assert "Claudexor model catalog" in prepared[1]["content"][1]["text"]


def test_our_local_lane_is_a_transport_fact_decided_by_lane(catalog, monkeypatch):
    """The " (local)" lane cannot carry image bytes because of OUR transport; the
    policy names that lane, and the capability reader returns no model verdict for it."""
    _, calls = catalog
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    prepared = prepare_messages_for_send(image_messages(), routing=VisionRoutingContext(
        MODEL + " (local)", object(), {}))
    assert "our local llama.cpp transport lane cannot carry images" in prepared[1]["content"][1]["text"]
    assert provider_models.supports_vision(MODEL + " (local)") is None
    assert calls == []


def test_caption_calls_use_vision_role_and_preserve_canonical_image(subscription_transport, catalog, monkeypatch):
    root, gateway, client = subscription_transport
    _, calls = catalog
    install_preset(monkeypatch)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "caption")
    messages = image_messages()
    original = deepcopy(messages)
    prepared = prepare_messages_for_send(messages, routing=VisionRoutingContext(
        MODEL, client, {}, drive_root=root, task_id="task-one"))
    assert "Ответ 🐍" in prepared[1]["content"][1]["text"]
    assert messages == original
    # The explicit vision slot is used without asking its catalog (owner decision);
    # the caption call itself still binds the vision role's account.
    assert calls == []
    assert gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "vision-b"}
    assert gateway.uploads[0][0]["messages"][-1]["content"][1]["type"] == "image_url"
    assert len(gateway.creates) == 1


def test_temporary_vision_role_uses_new_model_account_only(catalog, tmp_path):
    _, calls = catalog
    changed = "claudexor::another=new-image-model"
    with task_model_wait_scope(task={"id": "image-task", "_attempt": 1}, drive_root=tmp_path,
                               event_queue=None, worker_slot_held=True) as wait:
        wait.overrides["vision"] = {"model": changed, "model_account_override": "changed-profile", "use_local": False}
        assert vision._resolve_vlm_model(LLMClient(), MODEL) == changed
        from ouroboros.vision_routing import resolve_vision_caption_model
        assert resolve_vision_caption_model(SimpleNamespace(model=MODEL), LLMClient()) == changed
    # The owner's switch is an explicit choice of the vision route: it is used
    # without a catalog veto, so no other account's catalog is consulted either.
    assert calls == []


def test_browser_screenshot_keeps_subscription_image(catalog, tmp_path, monkeypatch):
    from ouroboros.tools.browser import _inject_native_screenshot
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    ctx = SimpleNamespace(active_model=MODEL, task_metadata={}, messages=[], drive_root=tmp_path)
    encoded = base64.b64encode(b"fixture-image").decode()
    _inject_native_screenshot(ctx, encoded)
    assert ctx.messages[-1]["content"][-1]["image_url"]["url"] == "data:image/png;base64," + encoded
    assert prepare_messages_for_send(ctx.messages, routing=VisionRoutingContext(MODEL, object(), {})) is ctx.messages


@pytest.mark.browser
@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_real_browser_screenshot_reaches_claudexor_payload_under_image_input(
        subscription_transport, catalog, monkeypatch, tmp_path, engine):
    """Seam row 4: a REAL browser screenshot travels to the real gateway payload.

    Real Playwright screenshot (same readonly-startup discipline as
    test_browser_page_wait) → attach_local_image_to_context → send-time routing
    under imageInput=true (inline gate passes, no re-encode here: attach already
    produced the payload) → call_llm_with_retry → the fake-but-real gateway
    upload. The payload carries the image block itself; only host metadata
    (_caption/_source_path) is stripped by the transport.
    """
    pytest.importorskip("playwright.sync_api")
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.loop_llm_call import call_llm_with_retry
    from ouroboros.tools import browser as browser_tools
    from ouroboros.tools.registry import ToolContext

    root, gateway, client = subscription_transport
    state, _ = catalog
    state["image_input"] = True
    install_preset(monkeypatch)
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")

    # Direct internal calls keep the whole multi-step browser session on ONE
    # thread, the discipline the loop's thread-sticky stateful executor owns
    # in production (same pattern as the browser isolation tests); the URL and
    # action policy inside both functions stays fully live.
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data", workspace_root=str(tmp_path),
                      task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    page = tmp_path / "index.html"
    page.write_text('<!doctype html><html><head><title>Seam proof</title></head>'
                    '<body style="font:24px sans-serif;background:#edf4ff"><h1>Seam proof</h1></body></html>',
                    encoding="utf-8")
    try:
        text = browser_tools._browse_page(ctx, page.as_uri(), engine=engine)
        if "not already installed" in text:
            # Readonly startup must use an installed browser; never install here.
            if engine in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(","):
                pytest.fail(text)
            pytest.skip(text)
        assert "Seam proof" in text
        shot = browser_tools._browser_action(ctx, "screenshot")
        assert "Screenshot captured" in shot
        encoded = ctx.browser_state.last_screenshot_b64
        assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")

        # The attach path itself is part of the seam: the real captured PNG is
        # loaded, re-encoded and durable-copied like any local image file.
        import pathlib
        shot_path = pathlib.Path(str(ctx.drive_root)) / "uploads" / "views" / "seam.png"
        shot_path.parent.mkdir(parents=True, exist_ok=True)
        shot_path.write_bytes(base64.b64decode(encoded))
        ctx.messages = []
        ok, _msg = vision.attach_local_image_to_context(ctx, str(shot_path))
        assert ok
        block = ctx.messages[-1]["content"][-1]
        assert block["type"] == "image_url" and block["image_url"]["url"].startswith("data:image/")
        original = deepcopy(ctx.messages)

        prepared = prepare_messages_for_send(ctx.messages, routing=VisionRoutingContext(MODEL, client, {}))
        assert prepared is ctx.messages  # imageInput=true keeps the block inline
        (root / "logs").mkdir(parents=True, exist_ok=True)
        answer, _ = call_llm_with_retry(client, deepcopy(ctx.messages), MODEL, [], "high", 1,
            root / "logs", "task-one", 1, None, {}, model_role="main")
        assert answer["content"] == "Ответ 🐍"
        assert ctx.messages == original  # canonical input never mutated
        sent = gateway.uploads[0][0]["messages"]
        payload_block = next(b for m in sent if isinstance(m.get("content"), list)
                             for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url")
        assert payload_block["image_url"]["url"] == block["image_url"]["url"]
        assert "_caption" not in payload_block and "_source_path" not in payload_block
    finally:
        browser_tools.cleanup_browser(ctx)


@pytest.mark.serial
@pytest.mark.parametrize("name", ["vlm_query", "analyze_screenshot"])
def test_registered_vision_tool_reaches_real_child_from_preset(child_fixture, catalog, tmp_path, monkeypatch, name):
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_vision_model_wait import MODEL as child_model

    _, events, root = child_fixture
    install_preset(monkeypatch, child_model)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=root)
    registry._ctx.task_id = "image-task"
    registry._ctx.task_attempt = 1
    # A URL stays remote; a valid 1x1 PNG passes the shared byte check unchanged
    # (invalid base64 is refused before any route). The child does not inspect pixels.
    registry._ctx.browser_state.last_screenshot_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
    args = {"prompt": "Describe only this image"}
    if name == "vlm_query":
        args["image_url"] = "https://example.invalid/fixture.png"
    result = registry.execute_result(name, args)
    assert result.text == "pixels read exactly once 🐍"
    rows = _events(events)
    assert sum(row["kind"] == "generation" for row in rows) == 1
    payload = next(row["payload"] for row in rows if row["kind"] == "upload")
    assert payload["account"] == {"mode": "pin", "profileId": "vision-b"}
    assert payload["messages"][-1]["content"][1]["type"] == "image_url"
