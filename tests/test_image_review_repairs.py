"""Consumer regressions for image-only ingress and host-owned VLM disclosures."""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.test_live_image_delivery import decode_block, image_blocks, pixels
from tests.test_model_wait import live_wait as live_wait, setup as setup
from tests.test_vision_model_wait import child_fixture as child_fixture


@pytest.mark.parametrize("kind", ["image", "file", "mixed"])
@pytest.mark.parametrize("delivery", ["named", "unnamed", "write_then_retry"])
def test_host_attachment_only_accept_rejoin_and_real_server_dequeue(tmp_path, monkeypatch, kind, delivery):
    import server
    from ouroboros.utils import iter_jsonl_objects
    from supervisor import message_bus
    from tests.test_chat_inject_attachments import _client, _skill_file

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    routed = []
    monkeypatch.setattr(server, "_route_owner_message", lambda _bridge, _ctx, row: routed.append(row))
    ctx = SimpleNamespace(load_state=lambda: {"owner_id": 42},
                          update_state=lambda mutator: mutator({"owner_id": 42}))
    client = _client(tmp_path, bridge)
    # The suffix/claimed MIME cannot decide the placeholder.
    sources = [_skill_file(tmp_path, "file.png", b"ordinary file")]
    if kind != "file":
        sources.append(_skill_file(tmp_path, "image.bin", pixels()))
    if kind == "image":
        sources.pop(0)
    body = {"chat_id": 42, "user_id": 42, "text": "  ", "image_caption": "\t",
            "attachments": [{"path": str(source), "mime": "image/png"} for source in sources]}
    if delivery != "unnamed":
        body["client_message_id"] = "skill-file-image"
    headers = {"X-Skill-Token": "token"}
    if delivery == "write_then_retry":
        original = message_bus.log_chat

        def fail_after_write(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("controlled failure after canonical acceptance")

        with monkeypatch.context() as scoped:
            scoped.setattr(message_bus, "log_chat", fail_after_write)
            assert client.post("/chat/inject", headers=headers, json=body).status_code == 500
        assert bridge._inbox.empty()
    response = client.post("/chat/inject", headers=headers, json=body)
    assert response.status_code == 202, response.text
    assert bridge._inbox.qsize() == 1
    if delivery != "unnamed":
        retry = client.post("/chat/inject", headers=headers, json=body)
        assert retry.status_code == 202 and retry.json()["rejoined"] is True
        assert bridge._inbox.qsize() == 1
    server._process_bridge_updates(bridge, 0, ctx)
    assert bridge._inbox.empty() and len(routed) == 1
    expected = "(file attached)" if kind == "file" else "(image attached)"
    rows = list(iter_jsonl_objects(tmp_path / "logs/chat.jsonl"))
    assert len(rows) == 1 and rows[0]["text"] == expected and rows[0]["text_placeholder"] is True
    assert routed[0]["log_text"] == expected
    assert routed[0]["origin_message_ref"]["text_sha256"] == __import__("hashlib").sha256(expected.encode()).hexdigest()
    uploads = routed[0]["task_metadata"]["chat_attachment_uploads"]
    assert [Path(item["path"]).read_bytes() for item in uploads] == [source.read_bytes() for source in sources]
    if delivery != "unnamed":
        assert client.post("/chat/inject", headers=headers, json=body).json()["rejoined"] is True
        sources[-1].write_bytes(b"changed content")
        assert client.post("/chat/inject", headers=headers, json=body).status_code == 409
        assert bridge._inbox.empty() and len(list(iter_jsonl_objects(tmp_path / "logs/chat.jsonl"))) == 1


def _registered_query(tmp_path, monkeypatch, client, tool, raw, model, *, real_child=False):
    from ouroboros.tools import vision, vision_process
    from tests.test_vision import _vision_registry

    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    monkeypatch.setattr(vision, "_get_llm_client", lambda: client)

    # Keep the registered handler and its waitable wrapper. Only the child transport
    # seam is in-process, so the real LLMClient and task wait controller are exercised.
    def child(**kwargs):
        kwargs.pop("child_timeout")
        kwargs.pop("subscription")
        return client.vision_query(**kwargs)

    if not real_child:
        monkeypatch.setattr(vision_process, "run_vision_child", child)
    args = {"prompt": "Inspect the pixels.", "model": model}
    if tool == "vlm_query":
        source = uploads / "input.image"
        source.write_bytes(raw)
        args["file_path"] = str(source)
    else:
        registry._ctx.browser_state.last_screenshot_b64 = base64.b64encode(raw).decode()
    return registry.execute_result(tool, args)


@pytest.mark.serial
@pytest.mark.parametrize("tool", ["vlm_query", "analyze_screenshot"])
def test_registered_vlm_common_note_survives_real_child_ipc(child_fixture, tmp_path, monkeypatch, tool):
    from ouroboros.llm import LLMClient
    from tests.test_vision_model_wait import MODEL, _events

    _state, events, _root = child_fixture
    result = _registered_query(tmp_path, monkeypatch, LLMClient(), tool, pixels("BMP"), MODEL, real_child=True)
    assert result.status == "ok", result.text
    assert result.text.count("Converted image/bmp to PNG") == 1
    assert "pixels read exactly once" in result.text
    uploads = [event["payload"] for event in _events(events) if event["kind"] == "upload"]
    assert len(uploads) == 1 and json.dumps(uploads).count("Converted image/bmp to PNG") == 1


@pytest.mark.parametrize("tool", ["vlm_query", "analyze_screenshot"])
def test_registered_vlm_reprepare_removes_old_route_note(live_wait, monkeypatch, tool):
    root, _transport, client, controller, _events, _decide = live_wait
    calls = []

    def chat(**kwargs):
        calls.append(kwargs["messages"])
        # The same callback used by wait/switch, in the reverse direction: the new
        # unknown route can carry the canonical width again, so the old limit expires.
        fresh = controller.reprepare("vision", {**kwargs, "model": "openai::unlisted-image-model"})
        calls.append(fresh["messages"])
        return {"content": "Model answer without disclosures."}, {}

    monkeypatch.setattr(client, "chat", chat)
    result = _registered_query(root, monkeypatch, client, tool, pixels("BMP", (9000, 24)),
                               "anthropic::unlisted-image-model")
    assert result.status == "ok", result.text
    assert json.dumps(calls[0]).count("Reduced dimensions") == 1
    assert "Reduced dimensions" not in json.dumps(calls[1]) and "Reduced dimensions" not in result.text
    for message in calls:
        assert json.dumps(message).count("Converted image/bmp to PNG") == 1
        assert json.dumps(message).count("Inspect the pixels.") == 1
    assert result.text.count("Converted image/bmp to PNG") == 1
    with Image.open(io.BytesIO(decode_block(image_blocks(calls[1])[0]))) as restored:
        assert restored.size == (9000, 24)


@pytest.mark.parametrize("tool", ["vlm_query", "analyze_screenshot"])
@pytest.mark.parametrize("kind", ["bmp", "partial"])
def test_registered_vlm_preserves_common_note_in_request_and_host_result(tmp_path, monkeypatch, tool, kind):
    from ouroboros.llm import LLMClient

    raw = pixels("BMP") if kind == "bmp" else pixels("JPEG", (100, 100))[:-20]
    note = "Converted image/bmp to PNG" if kind == "bmp" else "Partially recovered image"
    client, sent = LLMClient(api_key="fixture-no-network"), []

    def chat(**kwargs):
        sent.append(kwargs["messages"])
        return {"content": "Model answer without disclosures."}, {}

    monkeypatch.setattr(client, "chat", chat)
    result = _registered_query(tmp_path, monkeypatch, client, tool, raw, "openai::unlisted-image-model")
    assert result.status == "ok", result.text
    assert result.text.count(note) == 1 and "Model answer without disclosures." in result.text
    assert json.dumps(sent).count(note) == 1
    assert json.dumps(sent).count("Inspect the pixels.") == 1
    with Image.open(io.BytesIO(decode_block(image_blocks(sent[0])[0]))) as decoded:
        decoded.load()
        assert decoded.format == "PNG"


@pytest.mark.parametrize("tool", ["vlm_query", "analyze_screenshot", "caption"])
@pytest.mark.parametrize("no_pixels", [False, True])
def test_vlm_and_caption_quota_switch_notes_and_no_text_only_generation(live_wait, monkeypatch, tool, no_pixels):
    import requests
    from ouroboros import pricing, vision_routing as vr
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_model_wait import MODEL, _action_for, _refusal

    root, transport, client, controller, events, decide = live_wait
    transport.results, transport.dispatch = [_refusal()], ["not_started"]
    final_model = "anthropic::unlisted-image-model"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-no-network")
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_a, **_k: [])
    monkeypatch.setattr(vr, "_image_input_verdict", lambda *_a, **_k: None)

    def switch(*_a, **_k):
        event = next(row for row in reversed(list(events.queue)) if row.get("state") == "waiting")
        response = decide(_action_for(event, "switch", model=final_model,
            credential_profile_id="", use_local=False, persist_role=False))
        assert response.status_code == 202, response.body
        raise ClaudexorUnavailable("daemon_unreachable", "controlled catalog outage")

    monkeypatch.setattr(client, "claudexor_model_catalog", switch)
    sent = []

    def post(_session, _url, **kwargs):
        sent.append(kwargs["json"])
        response = requests.Response()
        response.status_code, response.url = 200, "https://api.anthropic.com/v1/messages"
        response._content = json.dumps({"content": [{"type": "text", "text": "Model answer."}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 2}}).encode()
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    raw = b"\x00\x00\x00\x18ftypheic" + bytes(64) if no_pixels else pixels("BMP", (9000, 24))
    # Explicitly exercise a recognized format without its optional local codec.
    if no_pixels:
        monkeypatch.setattr(Image, "MIME", {k: v for k, v in Image.MIME.items() if v != "image/heic"})
    if tool == "caption":
        monkeypatch.setattr(vr, "resolve_vision_caption_model", lambda *_a, **_k: MODEL)
        url = "data:image/heic;base64," if no_pixels else "data:image/bmp;base64,"
        caption, failure = vr._caption_for_block(
            {"type": "image_url", "image_url": {"url": url + base64.b64encode(raw).decode()}},
            ctx=SimpleNamespace(), llm=client, accumulated_usage={})
        assert bool(failure) is no_pixels and bool(caption) is not no_pixels
        result_text = failure or caption
    else:
        result = _registered_query(root, monkeypatch, client, tool, raw, MODEL)
        assert result.status == ("error" if no_pixels else "ok"), result.text
        if no_pixels:
            assert result.code == "VLM_ERROR"
        result_text = result.text
    assert len(transport.uploads) == 1
    first = json.dumps(transport.uploads[0][0]["messages"])
    assert "Reduced dimensions" not in first
    assert any(row.get("resolution") == "model_switched" for row in list(events.queue))
    if no_pixels:
        assert "VLM_NO_IMAGE_PIXELS" in result_text and "IMAGE_ROUTE_FORMAT_UNSUPPORTED" in result_text
        assert "Local codec unavailable" in result_text
        if tool != "caption":
            assert "Local codec unavailable" in first
        assert sent == [], "no paid text-only generation after all image pixels are withheld"
    else:
        assert len(sent) == 1, result_text
        for disclosure in ("Converted image/bmp to PNG", "Reduced dimensions"):
            assert json.dumps(sent).count(disclosure) == result_text.count(disclosure) == 1
        assert first.count("Converted image/bmp to PNG") == (0 if tool == "caption" else 1)
        source = next(part["source"] for msg in sent[0]["messages"] for part in msg["content"] if part["type"] == "image")
        with Image.open(io.BytesIO(base64.b64decode(source["data"]))) as image:
            assert max(image.size) == 8000


def test_main_keeps_good_pixels_and_words_when_one_image_is_unpreparable(monkeypatch):
    from ouroboros import vision_routing as vr
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(vr, "get_image_input_mode", lambda: "inline")
    bad = b"\x00\x00\x00\x18ftypheic" + bytes(64)
    monkeypatch.setattr(Image, "MIME", {k: v for k, v in Image.MIME.items() if v != "image/heic"})
    canonical = [{"role": "user", "content": [{"type": "text", "text": "Keep my words."}] + [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}}
        for raw, mime in [(pixels(), "image/png"), (bad, "image/heic")]]}]
    projected = vr.prepare_messages_for_send(canonical, routing=vr.VisionRoutingContext(
        "anthropic::unlisted-image-model", LLMClient(), {}))
    assert len(image_blocks(projected)) == 1 and decode_block(image_blocks(projected)[0]) == pixels()
    assert "Keep my words." in json.dumps(projected) and "IMAGE_ROUTE_FORMAT_UNSUPPORTED" in json.dumps(projected)
    assert len(image_blocks(canonical)) == 2


def test_retained_original_is_published_only_after_atomic_write_and_recovers(tmp_path, monkeypatch):
    from hashlib import sha256
    from ouroboros import utils
    from ouroboros.image_preparation import retain_original

    raw = pixels()
    final = tmp_path / f"{sha256(raw).hexdigest()}_original.png"
    write = Path.write_bytes
    observed = []

    def partial_then_fail(path, data):
        assert path.parent == final.parent and path != final
        write(path, data[:10])
        observed.append(final.exists())
        raise OSError("controlled disk failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "write_bytes", partial_then_fail)
        with pytest.raises(OSError, match="controlled disk failure"):
            retain_original(tmp_path, raw, "original.png")
    assert observed == [False] and not final.exists() and not list(tmp_path.iterdir())
    replace = utils.replace_atomic

    def check_complete(source, target):
        assert source.read_bytes() == raw and target == final and not final.exists()
        replace(source, target)

    monkeypatch.setattr(utils, "replace_atomic", check_complete)
    assert retain_original(tmp_path, raw, "original.png").read_bytes() == raw


def test_real_palette_alpha_tiff_keeps_pixels_in_png_and_retained_original(tmp_path, monkeypatch):
    from tests.test_vision import _vision_registry

    original = Image.new("PA", (3, 2))
    original.putpalette([12, 80, 150] + [0] * 765)
    for x, alpha in enumerate((0, 37, 255)):
        original.putpixel((x, 0), (0, alpha))
    raw = io.BytesIO()
    original.save(raw, format="TIFF")
    with Image.open(io.BytesIO(raw.getvalue())) as decoded:
        assert decoded.mode == "PA", "real decoder fixture must reach the palette+alpha branch"
        expected = decoded.convert("RGBA").tobytes()
    registry, uploads = _vision_registry(tmp_path, monkeypatch)
    source = uploads / "palette-alpha.tiff"
    source.write_bytes(raw.getvalue())
    result = registry.execute_result("view_image", {"path": str(source)})
    assert result.status == "ok", result.text
    block = image_blocks(registry._ctx.messages)[0]
    with Image.open(io.BytesIO(decode_block(block))) as prepared:
        assert prepared.mode == "RGBA" and prepared.tobytes() == expected
    assert Path(block["_source_path"]).read_bytes() == raw.getvalue()
