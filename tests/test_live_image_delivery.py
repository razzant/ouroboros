"""Owner image pixels survive real staging/drain and final wire serialization."""
import base64
import copy
from hashlib import sha256
import io
import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest
from PIL import Image

from ouroboros import vision_routing as vr
from ouroboros.artifacts import stage_task_attachments
from ouroboros.context import build_user_content
from ouroboros.image_preparation import prepare_image_bytes
from ouroboros.llm import LLMClient
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.owner_mailbox import write_owner_message


def pixels(fmt="PNG", size=(41, 27), mode="RGB"):
    image = Image.new(mode, size, (40, 90, 130, 80) if mode == "RGBA" else (40, 90, 130))
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


def image_blocks(messages):
    return [block for message in messages for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "image_url"]


def decode_block(block):
    return base64.b64decode(block["image_url"]["url"].split(",", 1)[1])


def wire(client, messages, model="openai::unlisted-image-model"):
    target = client._resolve_remote_target(model)
    if target["provider"] == "anthropic":
        return client._build_remote_candidate(target, messages, "high", 128, "auto", None, None)
    return client._build_remote_kwargs(target, messages, "high", 128, "auto", None, None,
                                       skip_capability_fetch=True)


@pytest.mark.parametrize("mode", ["auto", "inline", "caption", "off"])
@pytest.mark.parametrize("producer", ["initial", "direct", "mailbox"])
def test_owner_delivery_reaches_serialized_input(tmp_path, monkeypatch, mode, producer):
    from ouroboros.agent import OuroborosAgent
    from ouroboros import artifacts

    # Force the complete reference: the image is beyond the inline manifest view.
    monkeypatch.setattr(artifacts, "_MAX_STAGED_ATTACHMENTS", 1)
    drive, task_id = tmp_path / "drive", "running-image-task"
    source, text_file = tmp_path / "owner.png", tmp_path / "notes.txt"
    source.write_bytes(pixels())
    text_file.write_text("owner notes")
    manifest = stage_task_attachments(drive, task_id, [{"path": str(text_file)}, {"path": str(source)}])
    incoming, messages, seen = queue.Queue(), [], set()
    owner = SimpleNamespace(task_attempt=1)
    if producer == "initial":
        projection = artifacts.attachment_manifest_projection(drive, task_id, manifest)
        assert "attachment_manifest_ref" in projection
        messages = [{"role": "user", "content": build_user_content({
            "id": task_id, "drive_root": drive, "text": "inspect owner image", "task_contract": projection})}]
    elif producer == "direct":
        agent = object.__new__(OuroborosAgent)
        agent._incoming_messages = incoming
        agent.inject_message("inspect owner image", (base64.b64encode(source.read_bytes()).decode(), "image/png"))
        _drain_incoming_messages(messages, incoming, drive, task_id, None, seen, owner)
    else:
        assert write_owner_message(drive, "inspect owner image", task_id, msg_id="image-entry",
                                   attachment_manifest=manifest, client_message_id="owner-message")
        _drain_incoming_messages(messages, incoming, drive, task_id, None, seen, owner)
        delivered = copy.deepcopy(messages)
        _drain_incoming_messages(messages, incoming, drive, task_id, None, seen, owner)
        assert messages == delivered, "mailbox delivery is acknowledged only once"
        assert owner._owner_directives[0]["source"] == "owner_mailbox"
        assert image_blocks([{"content": owner._owner_directives[0]["content"]}])
        # An ordinary subsequent entry must not re-inline the task's earlier images.
        assert write_owner_message(drive, "a later note", task_id, msg_id="text-entry")
        _drain_incoming_messages(messages, incoming, drive, task_id, None, seen, owner)
    original = copy.deepcopy(messages)
    assert len(image_blocks(messages)) == 1
    original_path = Path(image_blocks(messages)[0]["_source_path"])
    assert original_path.read_bytes() == source.read_bytes()
    calls = []
    client = LLMClient(api_key="test-no-network")
    client.vision_query = lambda _prompt, images, **_kwargs: (calls.append(images) or "visible owner pixels", {})
    monkeypatch.setattr(vr, "get_image_input_mode", lambda: mode)
    monkeypatch.setattr(vr, "_image_input_verdict", lambda *_a, **_k: None)
    monkeypatch.setattr(vr, "resolve_vision_caption_model", lambda *_a, **_k: "openai::caption-model")
    projected = vr.prepare_messages_for_send(messages, routing=vr.VisionRoutingContext(
        "openai::unlisted-image-model", client, {}))
    payload = json.loads(json.dumps(wire(client, projected)))
    serialized = json.dumps(payload)
    assert bool(image_blocks(payload["messages"])) is (mode in {"auto", "inline"})
    assert bool(calls) is (mode == "caption")
    if mode == "caption":
        assert "visible owner pixels" in serialized
    if mode == "off":
        assert "Off" in serialized
    assert "_source_path" not in serialized
    assert messages == original
    assert original_path.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("fmt,size,mode", [("PNG", (41, 27), "RGB"), ("PNG", (9000, 24), "RGB"),
                                           ("BMP", (41, 27), "RGB"), ("PNG", (41, 27), "RGBA")])
@pytest.mark.parametrize("explicit", [False, True])
def test_automatic_and_explicit_preparation_preserve_detail_and_original(tmp_path, monkeypatch, fmt, size, mode, explicit):
    raw = pixels(fmt, size, mode)
    if explicit:
        from tests.test_vision import _vision_registry
        registry, uploads = _vision_registry(tmp_path, monkeypatch)
        path = uploads / ("owner." + fmt.lower())
        path.write_bytes(raw)
        result = registry.execute_result("view_image", {"path": str(path)})
        assert result.status == "ok", result.text
        messages = registry._ctx.messages
        retained = Path(image_blocks(messages)[0]["_source_path"])
        path.write_bytes(b"source later replaced")
    else:
        path = tmp_path / ("owner." + fmt.lower())
        path.write_bytes(raw)
        manifest = stage_task_attachments(tmp_path, "owner-task", [{"path": str(path)}])
        messages = [{"content": build_user_content({"id": "owner-task", "drive_root": tmp_path,
                                                    "attachment_images": manifest})}]
        retained = Path(image_blocks(messages)[0]["_source_path"])
    assert retained.read_bytes() == raw
    block = image_blocks(messages)[0]
    with Image.open(io.BytesIO(decode_block(block))) as decoded:
        assert decoded.size == size
        if mode == "RGBA":
            assert decoded.getpixel((0, 0))[3] == 80
        if fmt == "BMP":
            assert decoded.format == "PNG"
            assert "Converted image/bmp to PNG" in json.dumps(messages)
    if fmt == "PNG":
        assert decode_block(block) == raw


def test_partial_jpeg_is_reencoded_and_corrupt_png_is_explained(tmp_path):
    original = pixels("JPEG", (100, 100))[:-20]
    prepared = prepare_image_bytes(original, max_bytes=8 * 1024 * 1024)
    assert "Partially recovered" in prepared.note
    assert prepared.data != original
    with Image.open(io.BytesIO(prepared.data)) as repaired:
        repaired.load()  # usable strict decode, not the same corrupt upload
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64))
    manifest = stage_task_attachments(tmp_path, "broken", [{"path": str(path)}])
    content = build_user_content({"id": "broken", "drive_root": tmp_path, "attachment_images": manifest})
    assert not image_blocks([{"content": content}])
    assert "IMAGE_UNDECODABLE" in json.dumps(content)
    assert "Original:" in json.dumps(content)
    assert Path(manifest[0]["abs_path"]).read_bytes() == path.read_bytes()


def test_decoder_absence_preserves_unknown_route_bytes(monkeypatch):
    import builtins
    raw = pixels("BMP")
    original_import = builtins.__import__

    def without_pillow(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("decoder unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pillow)
    prepared = prepare_image_bytes(raw, max_bytes=8 * 1024 * 1024)
    assert prepared.data == raw and prepared.mime == "image/bmp"
    assert "without validation or conversion" in prepared.note


def test_route_limit_precedes_physical_identity_and_refusal_and_reprepare(monkeypatch):
    from tests.test_image_refusal_retry import _attempt_request

    raw = pixels(size=(9000, 24))
    original_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": original_url}}]}]
    original = copy.deepcopy(messages)
    monkeypatch.setattr(vr, "get_image_input_mode", lambda: "inline")
    client, usage = LLMClient(api_key="test-no-network"), {}
    model = "anthropic::unlisted-image-model"
    routing = vr.VisionRoutingContext(model, client, usage)
    projected = vr.prepare_messages_for_send(messages, routing=routing)
    block = image_blocks(projected)[0]
    with Image.open(io.BytesIO(decode_block(block))) as decoded:
        assert max(decoded.size) == 8000
    url = block["image_url"]["url"]
    digest = sha256(url.encode()).hexdigest()
    assert url != original_url
    payload = wire(client, projected, model)
    assert "_original_image_url" not in json.dumps(payload)
    request = _attempt_request(client._resolve_remote_target(model), payload)
    assert vr.candidate_images(request.candidate_raw_sha256) == {digest}
    vr.record_image_refusal(usage, model, [digest], {"status": 400, "message": "refused image"})
    retry = vr.prepare_messages_for_send(messages, routing=routing)
    assert not image_blocks(retry)
    assert "refused image" in json.dumps(retry)
    other = vr.prepare_messages_for_send(messages, routing=vr.VisionRoutingContext("openai::other", client, usage))
    assert image_blocks(other)[0]["image_url"]["url"] == original_url
    assert messages == original


def test_caption_hashes_its_own_prepared_pixels(monkeypatch):
    raw = pixels(size=(9000, 24))
    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}}
    calls, usage = [], {}
    client = LLMClient(api_key="test-no-network")
    client.vision_query = lambda _prompt, images, **_kwargs: (calls.append(images) or "caption result", {})
    monkeypatch.setattr(vr, "resolve_vision_caption_model", lambda *_a, **_k: "anthropic::caption")
    for _ in range(2):
        result, error = vr._caption_for_block(block, ctx=SimpleNamespace(), llm=client, accumulated_usage=usage)
        assert "caption result" in result and not error
    assert len(calls) == 1
    sent_url = calls[0][0]["url"]
    with Image.open(io.BytesIO(base64.b64decode(sent_url.split(",", 1)[1]))) as decoded:
        assert max(decoded.size) == 8000
    key = next(iter(usage["_vision_caption_memo"]))
    assert key.startswith(sha256(sent_url.encode()).hexdigest())


def damaged(kind):
    if kind == "uniform":  # a real, flat photo missing only the end-of-image marker
        return pixels("JPEG", (100, 100))[:-2]
    if kind == "partial":
        return pixels("JPEG", (100, 100))[:-20]
    if kind == "partial-png":  # a screenshot cut short: the rows it kept are real pixels
        raw = detailed_png()
        return raw[:len(raw) // 2]
    if kind == "blank":  # a tolerant decoder can return fill; its provenance is uncertain
        raw = detailed_png()
        start = raw.index(b"IDAT") + 4
        return raw[:start] + bytes(len(raw) - start)
    return b"\x89PNG\r\n\x1a\n" + bytes(64)


def detailed_png(size=(120, 90)):
    data = bytes((index * 37 + (index >> 7) * 11) % 256 for index in range(size[0] * size[1] * 3))
    image = Image.frombytes("RGB", size, data)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("kind", ["partial", "partial-png", "uniform", "corrupt", "blank"])
def test_damaged_input_consumers_keep_original_and_explain_pixels(tmp_path, monkeypatch, explicit, kind):
    raw = damaged(kind)
    if explicit:
        from tests.test_vision import _vision_registry
        registry, uploads = _vision_registry(tmp_path, monkeypatch)
        path = uploads / ("damaged.jpg" if kind == "partial" else "damaged.png")
        path.write_bytes(raw)
        result = registry.execute_result("view_image", {"path": str(path)})
        messages, words = registry._ctx.messages, result.text
        retained = list((tmp_path / "uploads" / "views").iterdir())
        assert len(retained) == 1 and retained[0].read_bytes() == raw
    else:
        path = tmp_path / ("damaged.jpg" if kind == "partial" else "damaged.png")
        path.write_bytes(raw)
        manifest = stage_task_attachments(tmp_path, "damage", [{"path": str(path)}])
        messages = [{"content": build_user_content({"id": "damage", "drive_root": tmp_path,
                                                    "attachment_images": manifest})}]
        words = json.dumps(messages)
        assert Path(manifest[0]["abs_path"]).read_bytes() == raw
    if kind != "corrupt":
        assert "Partially recovered" in words
        assert "decoder fill" in words and "uncertain" in words
        assert decode_block(image_blocks(messages)[0]) != raw
        with Image.open(io.BytesIO(decode_block(image_blocks(messages)[0]))) as readable:
            readable.load()
            if kind == "uniform":
                with Image.open(io.BytesIO(raw + b"\xff\xd9")) as intact:
                    assert readable.size == intact.size
                    assert readable.convert("RGB").tobytes() == intact.convert("RGB").tobytes()
    else:
        assert "IMAGE_UNDECODABLE" in words
        assert not image_blocks(messages)


def test_vlm_route_change_reprepares_original_before_serialization():
    from ouroboros.vision_image_limits import prepare_query_images

    raw = pixels(size=(9000, 24))
    original = [{"base64": base64.b64encode(raw).decode(), "mime": "image/png"}]
    bounded = prepare_query_images(original, "anthropic::vision")
    client, captured = LLMClient(api_key="test-no-network"), []
    client.chat = lambda **kwargs: (captured.append(wire(client, kwargs["messages"], kwargs["model"])) or {"content": "ok"}, {})
    client.vision_query("look", bounded, model="openai::different-route")
    restored = image_blocks(captured[0]["messages"])[0]
    assert decode_block(restored) == raw


def test_strict_decode_stays_strict_when_another_reader_left_the_switch_tolerant(monkeypatch):
    from PIL import ImageFile

    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    intact, cut = detailed_png(), damaged("partial-png")
    assert prepare_image_bytes(intact, max_bytes=8 << 20).data == intact
    recovered = prepare_image_bytes(cut, max_bytes=8 << 20)
    assert "Partially recovered" in recovered.note and recovered.data != cut
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True, "the caller's switch is restored"


def test_gain_map_jpeg_is_a_photo_not_an_animation(tmp_path):
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 10)).save(out, format="MPO", save_all=True,
                                                    append_images=[Image.new("RGB", (32, 24))])
    raw = out.getvalue()
    with Image.open(io.BytesIO(raw)) as probe:
        assert probe.format == "MPO" and probe.n_frames == 2
    path = tmp_path / "phone.jpg"
    path.write_bytes(raw)
    manifest = stage_task_attachments(tmp_path, "photo", [{"path": str(path)}])
    content = build_user_content({"id": "photo", "drive_root": tmp_path, "attachment_images": manifest})
    block = image_blocks([{"content": content}])[0]
    assert decode_block(block) == raw and block["image_url"]["url"].startswith("data:image/jpeg;")
    assert "frame" not in json.dumps(content) and "Converted" not in json.dumps(content)
    animated = io.BytesIO()
    Image.new("P", (8, 8), 1).save(animated, format="GIF", save_all=True, append_images=[Image.new("P", (8, 8), 2)])
    assert prepare_image_bytes(animated.getvalue(), max_bytes=8 << 20).data == animated.getvalue()


def test_an_unidentified_format_is_a_codec_gap_and_a_known_one_is_damage():
    with pytest.raises(ValueError, match="IMAGE_FORMAT_UNSUPPORTED.*not shown to be damaged"):
        prepare_image_bytes(b"unrecognized image format", max_bytes=8 << 20)
    with pytest.raises(ValueError, match="IMAGE_UNDECODABLE"):
        prepare_image_bytes(damaged("corrupt"), max_bytes=8 << 20)


@pytest.mark.parametrize("explicit", [False, True])
def test_recognized_heic_without_codec_reaches_unknown_route_with_disclosure(tmp_path, monkeypatch, explicit):
    from ouroboros import image_preparation, chat_uploads
    from PIL import UnidentifiedImageError

    # A recognized container header proves its media kind, not valid pixels.
    raw = b"\x00\x00\x00\x18ftypheic" + bytes(64)
    assert chat_uploads.detect_media(raw, "photo.heic") == ("image/heic", "image")
    monkeypatch.setattr(image_preparation, "_decode", lambda *_a, **_k: (_ for _ in ()).throw(UnidentifiedImageError()))
    monkeypatch.setattr(Image, "MIME", {k: v for k, v in Image.MIME.items() if v != "image/heic"})
    if explicit:
        from tests.test_vision import _vision_registry
        registry, uploads = _vision_registry(tmp_path, monkeypatch)
        source = uploads / "photo.heic"
        source.write_bytes(raw)
        result = registry.execute_result("view_image", {"path": str(source)})
        assert result.status == "ok", result.text
        messages = registry._ctx.messages
    else:
        source = tmp_path / "photo.heic"
        source.write_bytes(raw)
        manifest = stage_task_attachments(tmp_path, "heic", [{"path": str(source)}])
        messages = [{"role": "user", "content": build_user_content({
            "id": "heic", "drive_root": tmp_path, "attachment_images": manifest})}]
    original = copy.deepcopy(messages)
    block = image_blocks(messages)[0]
    assert Path(block["_source_path"]).read_bytes() == raw
    assert "without validation or conversion" in json.dumps(messages)
    monkeypatch.setattr(vr, "get_image_input_mode", lambda: "inline")
    client = LLMClient(api_key="test-no-network")
    projected = vr.prepare_messages_for_send(messages, routing=vr.VisionRoutingContext("openai::unknown", client, {}))
    assert decode_block(image_blocks(wire(client, projected)["messages"])[0]) == raw
    bounded = vr.prepare_messages_for_send(messages, routing=vr.VisionRoutingContext("anthropic::unknown", client, {}))
    assert not image_blocks(bounded) and "IMAGE_ROUTE_FORMAT_UNSUPPORTED" in json.dumps(bounded)
    assert "Original" in json.dumps(bounded) and messages == original
    with pytest.raises(ValueError, match="VLM_IMAGE_TOO_LARGE"):
        prepare_image_bytes(raw, max_bytes=len(raw) - 1)


def _without_pillow(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def without(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("decoder unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without)


@pytest.mark.parametrize("oversized", [False, True])
def test_decoder_absence_on_the_bounded_route_never_invents_a_size_failure(monkeypatch, oversized):
    raw = pixels() if not oversized else b"\x89PNG\r\n\x1a\n" + bytes(7_600_000)
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    monkeypatch.setattr(vr, "get_image_input_mode", lambda: "inline")
    client = LLMClient(api_key="test-no-network")
    _without_pillow(monkeypatch)
    projected = vr.prepare_messages_for_send(messages, routing=vr.VisionRoutingContext(
        "anthropic::unlisted-image-model", client, {}))
    words = json.dumps(projected)
    if oversized:
        assert not image_blocks(projected) and "VLM_IMAGE_TOO_LARGE" in words
    else:
        assert image_blocks(projected)[0]["image_url"]["url"] == url and "TOO_LARGE" not in words


def test_one_unpreparable_image_fails_only_its_own_caption(monkeypatch):
    good = "data:image/png;base64," + base64.b64encode(pixels()).decode()
    broken = "data:image/png;base64," + base64.b64encode(damaged("corrupt")).decode()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}
                                             for url in (good, broken)]}]
    client, calls = LLMClient(api_key="test-no-network"), []
    client.vision_query = lambda _prompt, images, **_kwargs: (calls.append(images) or "a blue rectangle", {})
    monkeypatch.setattr(vr, "get_image_input_mode", lambda: "caption")
    monkeypatch.setattr(vr, "resolve_vision_caption_model", lambda *_a, **_k: "anthropic::caption")
    projected = vr.prepare_messages_for_send(messages, routing=vr.VisionRoutingContext(
        "openai::main-model", client, {}))
    texts = [block["text"] for block in projected[0]["content"]]
    assert texts[0] == "[image caption: a blue rectangle]" and len(calls) == 1
    assert texts[1].startswith("[image caption unavailable: image preparation failed: IMAGE_UNDECODABLE")
    assert "image preparation failed (" not in json.dumps(projected), "no whole-send withholding"


@pytest.mark.parametrize("model,routed", [("anthropic::vision", False), ("openai::vision", True)])
def test_a_route_whose_known_limits_cannot_carry_the_image_is_passed_over(monkeypatch, model, routed):
    from ouroboros.tools import vision

    images = [{"url": "data:image/png;base64," + base64.b64encode(damaged("corrupt")).decode()}]
    monkeypatch.setattr(vr, "_image_input_verdict", lambda *_a, **_k: None)
    chosen, passed_over = vision._vlm_route(object(), model, ctx=SimpleNamespace(), images=images)
    assert (chosen == model) is routed
    if not routed:
        assert "could not be prepared for the route's known limits: IMAGE_UNDECODABLE" in passed_over[0][1]


def test_owner_words_survive_an_unexpected_composition_failure(tmp_path, monkeypatch):
    from ouroboros import loop_round_limits

    drive, task_id = tmp_path / "drive", "words-task"
    source = tmp_path / "owner.png"
    source.write_bytes(pixels())
    manifest = stage_task_attachments(drive, task_id, [{"path": str(source)}])
    assert write_owner_message(drive, "look at this", task_id, msg_id="entry", attachment_manifest=manifest)

    def broken(*_args, **_kwargs):
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr(loop_round_limits, "build_incoming_user_content", broken)
    messages, owner, seen = [], SimpleNamespace(task_attempt=1), set()
    _drain_incoming_messages(messages, queue.Queue(), drive, task_id, None, seen, owner)
    delivered = json.dumps(messages)
    assert "look at this" in delivered and "attached images were not composed: RuntimeError" in delivered
    before = copy.deepcopy(messages)
    _drain_incoming_messages(messages, queue.Queue(), drive, task_id, None, seen, owner)
    assert messages == before, "the entry was acknowledged once, not redelivered"


def test_a_reused_original_survives_the_views_age_sweep(tmp_path):
    import os
    from ouroboros.image_preparation import retain_original
    from ouroboros.server_maintenance import prune_agent_media_uploads

    views = tmp_path / "uploads" / "views"
    stale = retain_original(views, b"first view", "a.png")
    reused = retain_original(views, b"viewed again", "b.png")
    month_ago = reused.stat().st_mtime - 40 * 86400
    for path in (stale, reused):
        os.utime(path, (month_ago, month_ago))
    assert retain_original(views, b"viewed again", "b.png") == reused
    prune_agent_media_uploads(tmp_path, retention_days=30)
    assert reused.read_bytes() == b"viewed again" and not stale.exists()


# The wait/decision and chat paths are real; only engine/HTTP transport is controlled.
from tests.test_model_wait import live_wait as live_wait, setup as setup  # noqa: E402


@pytest.mark.parametrize('refused', [False, True])
def test_caption_quota_wait_switch_reprepares_pixels_and_identity(live_wait, monkeypatch, refused):
    import requests
    from ouroboros import pricing
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_model_wait import MODEL, _action_for, _refusal

    root, transport, client, controller, events, decide = live_wait
    transport.results, transport.dispatch = [_refusal()], ['not_started']
    final_model = 'anthropic::unlisted-image-model'
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'fixture-no-network')
    monkeypatch.setattr(pricing, '_fetch_live_rows', lambda *_a, **_k: [])
    monkeypatch.setenv('OUROBOROS_MODEL_VISION', MODEL)
    monkeypatch.setattr(vr, '_image_input_verdict', lambda *_a, **_k: None)

    def switch(*_a, **_k):
        event = next(row for row in reversed(list(events.queue)) if row.get('state') == 'waiting')
        response = decide(_action_for(event, 'switch', model=final_model,
            credential_profile_id='', use_local=False, persist_role=False))
        assert response.status_code == 202, response.body
        raise ClaudexorUnavailable('daemon_unreachable', 'controlled catalog outage')

    monkeypatch.setattr(client, 'claudexor_model_catalog', switch)
    sent = []

    def post(_session, _url, **kwargs):
        sent.append(kwargs['json'])
        response = requests.Response()
        response.status_code = 400 if refused else 200
        response.url = 'https://api.anthropic.com/v1/messages'
        body = ({'error': {'type': 'invalid_request_error', 'message': 'controlled image refusal'}} if refused else
                {'content': [{'type': 'text', 'text': 'visible pixels'}], 'stop_reason': 'end_turn',
                 'usage': {'input_tokens': 20, 'output_tokens': 2}})
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.Session, 'post', post)
    raw = pixels(size=(9000, 24))
    url = 'data:image/png;base64,' + base64.b64encode(raw).decode()
    from ouroboros.vision_image_limits import prepare_image_block_for_route
    block, _ = prepare_image_block_for_route({'type': 'image_url', 'image_url': {'url': url}}, final_model)
    assert block['_original_image_url'] == url, 'Main may already have prepared a bounded derivative'
    original, usage = copy.deepcopy(block), {}
    caption, failure = vr._caption_for_block(block, ctx=SimpleNamespace(), llm=client, accumulated_usage=usage)
    assert bool(failure) is refused
    assert bool(caption) is not refused
    assert len(transport.uploads) == len(sent) == 1
    assert decode_block(image_blocks(transport.uploads[0][0]['messages'])[0]) == raw
    source = next(part['source'] for msg in sent[0]['messages'] for part in msg['content'] if part['type'] == 'image')
    sent_url = 'data:' + source['media_type'] + ';base64,' + source['data']
    with Image.open(io.BytesIO(base64.b64decode(source['data']))) as image:
        assert max(image.size) == 8000
    digest = sha256(sent_url.encode()).hexdigest()
    assert sent_url != url and '_original_image_url' not in json.dumps(sent)
    assert block == original
    assert any(row.get('resolution') == 'model_switched' for row in list(events.queue))
    if refused:
        check = vr.refusal_check(usage, [digest])
        assert check(final_model) and not check(MODEL), (failure, usage)
        assert not vr.refusal_check(usage, [sha256(url.encode()).hexdigest()])(final_model)
        assert usage['_vision_caption_memo'] == {}
    else:
        assert list(usage['_vision_caption_memo']) == [f'{digest}|{final_model}|v1']
        assert 'Reduced dimensions' in caption
        repeated, failure = vr._caption_for_block(block, ctx=SimpleNamespace(), llm=client, accumulated_usage=usage)
        assert repeated == caption and not failure
        assert len(transport.uploads) == len(sent) == 1, 'the switched route reuses its exact-pixel caption'
