"""Send-time image routing for inline vision versus caption text."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import logging
import pathlib
from typing import Any, Dict, List

from ouroboros.config import get_image_input_mode, get_vision_caption_timeout_sec, get_vision_model, resolve_effort
from ouroboros.deadline_utils import owner_deadline_exhausted, transport_timeout_with_deadline
from ouroboros.observability import new_call_id, persist_call
from ouroboros.provider_models import supports_vision
from ouroboros.utils import emit_cognitive_operation_event

log = logging.getLogger(__name__)


_CAPTION_PROMPT = (
    "Describe this image in detail for a coding/research agent that may not see pixels. "
    "Be objective and include visible text, UI state, diagrams, layout, and salient details. "
    "Do not infer hidden facts."
)


def _vision_finalization_reserve() -> float:
    try:
        from ouroboros.config import get_finalization_grace_sec
        return float(get_finalization_grace_sec())
    except Exception:
        return 0.0


@dataclass
class VisionRoutingContext:
    model: str
    llm: Any
    accumulated_usage: Dict[str, Any]
    drive_root: pathlib.Path | None = None
    task_id: str = ""
    event_queue: Any = None
    use_local: bool = False
    task_attempt: Any = None
    deadline_ts: Any = None


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
        from ouroboros.config import get_light_model, parse_fallback_chain

        candidates.append(get_light_model())
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
    emit_cognitive_operation_event(
        event_queue,
        task_id=task_id,
        operation_id=call_id,
        phase="started",
        kind="vlm",
        task_attempt=getattr(ctx, "task_attempt", None),
    )
    # Receipts are BOOKKEEPING and live OUTSIDE the caption-producing try: a
    # persist_call failure used to jump into the failure arm below, REPLACE the
    # paid caption with a failure label and memoize it for the task.
    if drive_root is not None:
        try:
            prompt_ref = persist_call(
                drive_root,
                task_id=task_id,
                call_id=f"{call_id}_request",
                call_type="vision_caption_request",
                payload={"prompt": _CAPTION_PROMPT, "image_url": url, "model": model},
                manifest={"model": model},
            )
        except Exception:
            log.warning("vision caption request receipt failed", exc_info=True)
    try:
        reserve = _vision_finalization_reserve()
        if owner_deadline_exhausted(
            deadline_ts=getattr(ctx, "deadline_ts", None), reserve_sec=reserve,
        ):
            raise TimeoutError("owner deadline leaves no window for a vision caption")
        text, usage = llm.vision_query(
            _CAPTION_PROMPT,
            [{"url": url}],
            model=model,
            reasoning_effort=resolve_effort("task"),
            timeout=transport_timeout_with_deadline(
                get_vision_caption_timeout_sec(),
                deadline_ts=getattr(ctx, "deadline_ts", None),
                reserve_sec=reserve,
            ),
        )
    except Exception as exc:
        emit_cognitive_operation_event(
            event_queue,
            task_id=task_id,
            operation_id=call_id,
            phase="failed",
            kind="vlm",
            task_attempt=getattr(ctx, "task_attempt", None),
        )
        # NOT memoized: a memoized failure label used to block every retry for
        # this image for the rest of the task.
        return f"[image caption unavailable: {type(exc).__name__}: {exc}]"
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
        try:
            persist_call(
                drive_root,
                task_id=task_id,
                call_id=f"{call_id}_response",
                call_type="vision_caption_response",
                payload={"caption": caption, "usage": usage, "prompt_ref": prompt_ref},
                manifest={"model": model},
            )
        except Exception:
            log.warning("vision caption response receipt failed", exc_info=True)
    emit_cognitive_operation_event(
        event_queue,
        task_id=task_id,
        operation_id=call_id,
        phase="finished",
        kind="vlm",
        task_attempt=getattr(ctx, "task_attempt", None),
    )
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
