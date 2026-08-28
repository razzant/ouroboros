"""Send-time image routing for inline vision versus caption text."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import pathlib
from typing import Any, Dict, List

from ouroboros.config import get_image_input_mode, get_vision_caption_timeout_sec, get_vision_model, resolve_effort
from ouroboros.observability import new_call_id, persist_call
from ouroboros.provider_models import supports_vision
from ouroboros.usage_accounting import physical_provider_outcome_unknown


_CAPTION_PROMPT = (
    "Describe this image in detail for a coding/research agent that may not see pixels. "
    "Be objective and include visible text, UI state, diagrams, layout, and salient details. "
    "Do not infer hidden facts."
)


@dataclass
class VisionRoutingContext:
    model: str
    llm: Any
    accumulated_usage: Dict[str, Any]
    drive_root: pathlib.Path | None = None
    task_id: str = ""
    event_queue: Any = None
    use_local: bool = False


def resolve_vision_caption_model(ctx: Any, llm: Any, *, use_local: bool = False) -> str:
    import os

    explicit_raw = str(os.environ.get("OUROBOROS_MODEL_VISION", "") or "").strip()
    explicit = str(get_vision_model() or "").strip()
    if use_local and not explicit_raw:
        return ""
    if explicit and supports_vision(explicit):
        return explicit
    candidates = [
        str(getattr(ctx, "model", "") or "").strip(),
        str(getattr(ctx, "active_model", "") or getattr(ctx, "task_model_override", "") or "").strip(),
    ]
    try:
        from ouroboros.config import get_light_model, get_heavy_model, parse_fallback_chain

        candidates.extend([get_light_model(), get_heavy_model()])
        candidates.extend(parse_fallback_chain())
    except Exception:
        pass
    try:
        candidates.append(str(llm.default_model() or "").strip())
    except Exception:
        pass
    for candidate in candidates:
        if candidate and supports_vision(candidate):
            return candidate
    return ""


def _image_url_from_block(block: Dict[str, Any]) -> str:
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "")
    return str(block.get("url") or "")


def _caption_for_block(
    block: Dict[str, Any],
    *,
    ctx: Any,
    llm: Any,
    accumulated_usage: Dict[str, Any],
    drive_root: pathlib.Path | None = None,
    task_id: str = "",
    event_queue: Any = None,
) -> str:
    memo = accumulated_usage.setdefault("_vision_caption_memo", {})
    url = _image_url_from_block(block)
    model = resolve_vision_caption_model(ctx, llm, use_local=bool(getattr(ctx, "use_local", False)))
    url_digest = sha256(url.encode("utf-8", errors="replace")).hexdigest()
    key = f"{url_digest}|{model}|v1"
    if key in memo:
        return str(memo[key] or "")
    if not model or not url:
        return ""
    call_id = new_call_id("vision_caption")
    prompt_ref = {}
    try:
        if drive_root is not None:
            prompt_ref = persist_call(
                drive_root,
                task_id=task_id,
                call_id=f"{call_id}_request",
                call_type="vision_caption_request",
                payload={"prompt": _CAPTION_PROMPT, "image_url": url, "model": model},
                manifest={"model": model},
            )
        text, usage = llm.vision_query(
            _CAPTION_PROMPT,
            [{"url": url}],
            model=model,
            reasoning_effort=resolve_effort("task"),
            timeout=get_vision_caption_timeout_sec(),
        )
        try:
            from ouroboros.llm import add_usage

            add_usage(accumulated_usage, usage)
        except Exception:
            pass
        try:
            from ouroboros.pricing import emit_llm_usage_event

            emit_llm_usage_event(
                event_queue,
                task_id,
                model,
                usage,
                (
                    float(usage["cost"])
                    if isinstance(usage, dict) and usage.get("cost") is not None
                    else None
                ),
                category="task",
                source="vision_caption",
            )
        except Exception:
            pass
        caption = str(text or "").strip()
        if drive_root is not None:
            persist_call(
                drive_root,
                task_id=task_id,
                call_id=f"{call_id}_response",
                call_type="vision_caption_response",
                payload={"caption": caption, "usage": usage, "prompt_ref": prompt_ref},
                manifest={"model": model},
            )
    except Exception as exc:
        if physical_provider_outcome_unknown(exc):
            raise
        caption = f"[image caption unavailable: {type(exc).__name__}: {exc}]"
    memo[key] = caption
    return caption


def _usable_existing_caption(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Browser/view_image producers use bracketed labels for eviction/re-view hints
    # (e.g. "[browser screenshot ...]" / "[image: file.png]"), not visual captions.
    if text.startswith("[") and text.endswith("]"):
        return ""
    return text


def prepare_messages_for_send(
    messages: List[Dict[str, Any]],
    *,
    routing: VisionRoutingContext,
) -> List[Dict[str, Any]]:
    mode = get_image_input_mode()
    model_supports_inline = (not routing.use_local) and supports_vision(routing.model)
    if (mode == "inline" and model_supports_inline) or (mode == "auto" and model_supports_inline):
        return messages
    has_image = any(
        isinstance(msg.get("content"), list)
        and any(isinstance(block, dict) and str(block.get("type") or "") in {"image_url", "image"} for block in msg["content"])
        for msg in messages
        if isinstance(msg, dict)
    )
    if not has_image:
        return messages
    if mode == "off":
        rewrite_to_caption = False
    elif mode == "inline":
        rewrite_to_caption = False
    elif mode == "caption" or mode == "auto":
        rewrite_to_caption = True
    else:
        return messages

    changed = False
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for idx, block in enumerate(content):
            if not isinstance(block, dict) or str(block.get("type") or "") not in {"image_url", "image"}:
                continue
            caption = ""
            if rewrite_to_caption:
                existing_caption = _usable_existing_caption(str(block.get("_caption") or ""))
                caption = existing_caption or _caption_for_block(
                    block,
                    ctx=routing,
                    llm=routing.llm,
                    accumulated_usage=routing.accumulated_usage,
                    drive_root=routing.drive_root,
                    task_id=routing.task_id,
                    event_queue=routing.event_queue,
                )
            if caption:
                content[idx] = {"type": "text", "text": f"[image caption: {caption}]"}
            else:
                content[idx] = {"type": "text", "text": "[image omitted: image input disabled or no vision model available]"}
            changed = True
    return out if changed else messages
