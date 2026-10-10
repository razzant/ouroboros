"""Subscription images reach the ordinary Main and registered VLM entrypoints."""

import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import provider_models
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
    state = {"modalities": ["text", "image"], "error": None}

    def read(source, profile=None, *, requested_model=None):
        calls.append((source, profile, requested_model))
        if state["error"]:
            raise ClaudexorUnavailable(state["error"], "controlled metadata unavailable")
        return {"source": source, "credentialProfileId": profile or "auto-account",
                "accountFingerprint": "identity-" + (profile or "auto-account"),
                "models": [{"id": requested_model, "inputModalities": state["modalities"]}]}

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
    state["modalities"] = ["text"]
    assert provider_models.supports_vision(MODEL, model_role="vision") is False
    state["modalities"] = ["text", "image"]
    assert provider_models.supports_vision(MODEL, model_role="vision", model_account_override="") is True
    assert calls == [("codex", "main-a", "exact-model"), ("codex", "vision-b", "exact-model"),
                     ("codex", None, "exact-model")]
    # An account's catalog answer is read per call; it never becomes a stored route record.
    store = capability_evidence._load(capability_evidence.canonical_evidence_root())
    assert store["image_input"] == {}


@pytest.mark.parametrize("modalities", [None, []])
def test_absent_image_metadata_stays_unknown_and_does_not_erase_input(catalog, monkeypatch, modalities):
    state, _ = catalog
    state["modalities"] = modalities
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "auto")
    assert provider_models.supports_vision(MODEL, model_role="main") is None
    messages = image_messages()
    assert prepare_messages_for_send(messages, routing=VisionRoutingContext(MODEL, object(), {})) is messages


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
    state["modalities"] = ["text"]
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
