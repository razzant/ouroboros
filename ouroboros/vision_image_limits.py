"""Known image limits of the selected wire route, applied before image identity.

This is not a model-name capability table. Only the direct, fixed Anthropic API
endpoint below has a verified bound here. Compatible gateways, relays and native
subscription routes keep unknown limits unknown.

Source (checked 2026-10-09): https://platform.claude.com/docs/en/build-with-claude/vision
The direct API accepts JPEG/PNG/GIF/WebP, up to 10 MB base64 per image and 8000 pixels per side.
The host keeps 7.5 MB raw, and the existing live-image budget keeps Main below the
many-image threshold. A known unsupported format without local conversion becomes an
explicit ``IMAGE_ROUTE_FORMAT_UNSUPPORTED`` route note.

Limits apply to the send copy before image hashing. A derivative carries
``_original_image_url`` (stripped before the wire): the refused-image set, caption memo
and physical capture hash the pixels actually sent, while a route change, caption
candidate or model-wait reprepare starts from the canonical image. ``vision_query``
registers its canonical preparation around the waitable send, so a wait/switch rebuilds
pixels and caption memo/refusal identity for the actual route. A caption or VLM
candidate whose known limits cannot carry the image is passed over with that reason;
one block's preparation failure stays that block's caption failure or note. No tiling
is added; a provider may still resize internally.
"""
from __future__ import annotations

import base64
from typing import Any

from ouroboros.image_preparation import prepare_image_bytes


def prepare_image_block_for_route(block: dict, model: str) -> tuple[dict, str]:
    from ouroboros.llm import LLMClient
    from ouroboros.vision_routing import _image_url_from_block

    # A caption candidate starts from the canonical image, not a derivative for
    # the Main route which happened to refuse it.
    original_url = block.get("_original_image_url") or _image_url_from_block(block)
    source = block
    if block.get("_original_image_url"):
        source = {**block, "type": "image_url", "image_url": {"url": original_url}}
        source.pop("source", None)
        source.pop("_original_image_url", None)
    target = LLMClient()._resolve_remote_target(model)
    if (target.get("provider"), target.get("base_url")) != ("anthropic", "https://api.anthropic.com/v1"):
        return source, ""
    if not original_url.startswith("data:") or ";base64," not in original_url:
        return source, ""  # URL/file-id image bytes remain owned by that remote source.
    raw = base64.b64decode(original_url.split(";base64,", 1)[1], validate=True)
    prepared = prepare_image_bytes(raw, max_bytes=7_500_000, max_side=8000,
                                   reason="the direct Anthropic API image limit")
    if prepared.mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise ValueError(f"IMAGE_ROUTE_FORMAT_UNSUPPORTED: the direct Anthropic API requires JPEG, PNG, GIF "
                         f"or WebP; {prepared.mime} could not be converted locally. {prepared.note} Original retained.")
    url = f"data:{prepared.mime};base64,{base64.b64encode(prepared.data).decode('ascii')}"
    if url == original_url:
        return source, prepared.note
    changed = {**source, "type": "image_url", "image_url": {"url": url},
               "_original_image_url": original_url}
    changed.pop("source", None)
    return changed, prepared.note


def prepare_route_images(messages: list[dict], model: str) -> list[dict]:
    from ouroboros.vision_routing import _is_image

    result = []
    changed = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            result.append(message)
            continue
        parts = []
        for block in content:
            if not _is_image(block):
                parts.append(block)
                continue
            try:
                prepared, note = prepare_image_block_for_route(block, model)
            except (OSError, ValueError) as exc:
                prepared, note = {"type": "text", "text": f"[image pixels unavailable: {exc}]"}, ""
                if block.get("_source_path"):
                    prepared["text"] += f" Original: {block['_source_path']}"
            changed |= prepared is not block
            if note:
                changed = True
                parts.append({"type": "text", "text": note})
            parts.append(prepared)
        result.append({**message, "content": parts})
    return result if changed else messages


def prepare_query_images(images: Any, model: str) -> list[dict]:
    from ouroboros.vision_routing import _image_url_from_block

    result = []
    for image in images:
        if image.get("url"):
            url = image["url"]
        else:
            url = f"data:{image.get('mime', 'image/png')};base64,{image.get('base64', '')}"
        source = {"type": "image_url", "image_url": {"url": url}}
        if image.get("_original_image_url"):
            source["_original_image_url"] = image["_original_image_url"]
        prepared, note = prepare_image_block_for_route(source, model)
        prepared_url = _image_url_from_block(prepared)
        result.append(dict(image) if prepared_url == url else {
            "url": prepared_url, "_original_image_url": image.get("_original_image_url") or url,
            "note": " ".join(part for part in (image.get("note"), note) if part)})
    return result


def refusal_identity(error: BaseException, model: str, fallback: Any) -> tuple[str, Any, list]:
    """``(route, refusal, digests)`` of a failed ``vision_query``: the route actually asked and,
    for a completed image refusal, the pixels it was actually sent (capture, receipt, then
    ``fallback(route)``). Accounting may spell a model provider/model, which would resolve as
    OpenRouter; the wait controller's route key wins."""
    from ouroboros.vision_routing import candidate_images, completed_image_refusal

    receipt = getattr(error, "vision_query_receipt", None) or {}
    model = str(receipt.get("model") or (getattr(error, "model_role_route", None) or {}).get("model") or model)
    refusal = completed_image_refusal(error)
    if refusal is None:
        return model, None, []
    digests = candidate_images(getattr(getattr(error, "physical_attempt_capture", None), "candidate_raw_sha256", None))
    return model, refusal, digests if digests is not None else receipt.get("digests") or fallback(model)


def sent_caption_identity(usage: Any, model: str, key: str, note: str) -> tuple[str, str, str]:
    """``(route, memo key, note)`` of the pixels a possibly wait-switched caption call sent.

    A caption without exactly one image has no reusable pixel identity."""
    receipt = (usage if isinstance(usage, dict) else {}).get("_vision_query_receipt") or {}
    if not receipt:
        return model, key, note
    model, digests = str(receipt.get("model") or model), receipt.get("digests") or []
    return model, f"{digests[0]}|{model}|v1" if len(digests) == 1 else "", str(receipt.get("note") or "")


def query_image_messages(prompt: str, images: list[dict]) -> list[dict]:
    """Preserve the original input across a VLM wait that selects a new route."""
    from ouroboros.vision_routing import query_image_url

    content = [{"type": "text", "text": prompt}]
    for image in images:
        if "url" not in image and "base64" not in image:
            continue
        if image.get("note"):
            content.append({"type": "text", "text": image["note"]})
        block = {"type": "image_url", "image_url": {"url": query_image_url(image)}}
        if image.get("_original_image_url"):
            block["_original_image_url"] = image["_original_image_url"]
        content.append(block)
    return [{"role": "user", "content": content}]


def prepare_caption_image(block: dict, ctx: Any, llm: Any, usage: dict) -> tuple[str, str, str, str, str]:
    """``(model, url, digest, note, failure)``: each candidate is checked with its own pixels.

    A candidate whose known limits cannot carry the image is passed over; a failure
    for the chosen one is that block's caption failure, never the whole send's.
    """
    from ouroboros import vision_routing as vr

    def refused(model):
        try:
            prepared, _note = prepare_image_block_for_route(block, model)
        except (OSError, ValueError) as exc:
            return f"this image could not be prepared for the route's known limits: {exc}"
        check = vr.refusal_check(usage, [vr._image_digest(prepared)])
        return check(model) if check else ""

    model = vr.resolve_vision_caption_model(ctx, llm, use_local=bool(getattr(ctx, "use_local", False)),
                                            refused=refused)
    if not model:
        return "", "", "", "", ""
    try:
        prepared, note = prepare_image_block_for_route(block, model)
    except (OSError, ValueError) as exc:
        return "", "", "", "", f"image preparation failed: {exc}"
    url = vr._image_url_from_block(prepared)
    return model, url, vr._url_digest(url), note, ""
