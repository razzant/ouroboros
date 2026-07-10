"""Vision LLM tools for browser screenshots and uploaded images."""

from __future__ import annotations

import logging
import pathlib
import os
import json
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.config import resolve_effort
from ouroboros.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)


def _get_llm_client():
    """Lazy-import LLMClient to avoid circular imports."""
    from ouroboros.llm import LLMClient
    return LLMClient()


def _analyze_screenshot(ctx: ToolContext, prompt: str = "Describe what you see in this screenshot. Note any important UI elements, text, errors, or visual issues.", model: str = "") -> str:
    """Analyze the last browser screenshot via VLM."""
    b64 = ctx.browser_state.last_screenshot_b64
    if not b64:
        return (
            "⚠️ No screenshot available. "
            "First call browse_page(output='screenshot') or browser_action(action='screenshot')."
        )

    try:
        client = _get_llm_client()
        vlm_model = _resolve_vlm_model(client, model, ctx=ctx)
        if not vlm_model:
            return _VLM_NO_VISION_MODEL_MSG
        text, usage = _vision_query_with_timeout(
            client,
            prompt=prompt,
            images=[_image_payload_from_base64(b64, "image/png")],
            model=vlm_model,
            reasoning_effort=resolve_effort("task"),
            timeout=_VLM_HTTP_TIMEOUT_SEC,
        )

        _emit_usage(ctx, usage, vlm_model)

        return text or "(no response from VLM)"
    except Exception as e:
        log.warning("analyze_screenshot failed: %s", e, exc_info=True)
        return f"⚠️ VLM_ANALYSIS_FAILED: {e}"


_IMAGE_MAGIC: List[tuple] = [
    (b'\x89PNG\r\n\x1a\n', "image/png"),
    (b'\xff\xd8\xff', "image/jpeg"),
    (b'GIF87a', "image/gif"),
    (b'GIF89a', "image/gif"),
]
_IMAGE_WEBP_MAGIC = (b'RIFF', b'WEBP')
_VLM_MAX_FILE_BYTES = 20 * 1024 * 1024
_VLM_MAX_PROVIDER_BYTES = 6 * 1024 * 1024
_VLM_MAX_IMAGE_SIDE = 1600
_VLM_HTTP_TIMEOUT_SEC = 90.0


def _vision_query_with_timeout(client: Any, **kwargs: Any) -> tuple[str, dict]:
    """Run a VLM query behind a tracked, killable child process."""
    del client  # production path constructs the client in the tracked child.
    timeout = float(kwargs.get("timeout") or _VLM_HTTP_TIMEOUT_SEC)
    payload = dict(kwargs)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
        json.dump(payload, fh)
        payload_path = fh.name
    script = r"""
import json
import sys
import time
from ouroboros.llm import LLMClient

with open(sys.argv[1], encoding="utf-8") as fh:
    kwargs = json.load(fh)
sleep_for = float(kwargs.pop("_test_sleep_sec", 0) or 0)
if sleep_for > 0:
    time.sleep(sleep_for)
try:
    text, usage = LLMClient().vision_query(**kwargs)
except BaseException as exc:  # noqa: BLE001
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(1)
print(json.dumps({"ok": True, "text": text, "usage": usage}))
"""
    try:
        from ouroboros.tools.shell import _tracked_subprocess_run

        python_exe = sys.executable or os.environ.get("OUROBOROS_AGENT_PYTHON") or "python3"
        res = _tracked_subprocess_run(
            [python_exe, "-c", script, payload_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"VLM query exceeded {timeout:g}s wall-clock timeout") from exc
    finally:
        try:
            os.unlink(payload_path)
        except OSError:
            pass
    lines = [line for line in str(res.stdout or "").splitlines() if line.strip()]
    data = json.loads(lines[-1]) if lines else {}
    if res.returncode == 0 and data.get("ok"):
        return str(data.get("text") or ""), data.get("usage") if isinstance(data.get("usage"), dict) else {}
    error = data.get("error") or str(res.stderr or "").strip() or "VLM subprocess failed"
    raise RuntimeError(str(error))


def _path_is_under(path: "pathlib.Path", root: "pathlib.Path") -> bool:
    """Return True if a resolved path is root itself or a descendant."""
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _detect_image_mime_for_vlm(raw: bytes) -> str:
    """Return MIME type string or empty string if not a recognised image."""
    for magic, mime in _IMAGE_MAGIC:
        if raw[:len(magic)] == magic:
            return mime
    if raw[:4] == _IMAGE_WEBP_MAGIC[0] and raw[8:12] == _IMAGE_WEBP_MAGIC[1]:
        return "image/webp"
    return ""


def _downscale_image_for_vlm(raw: bytes, mime: str) -> Tuple[bytes, str]:
    """Cap very large image payloads before sending them to the VLM provider."""
    if len(raw) <= _VLM_MAX_PROVIDER_BYTES:
        try:
            from PIL import Image
            import io

            with Image.open(io.BytesIO(raw)) as img:
                if max(img.size) <= _VLM_MAX_IMAGE_SIDE:
                    return raw, mime
        except Exception:
            return raw, mime

    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if img.mode != "RGB":
                background = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.getchannel("A") if img.mode in {"RGBA", "LA"} else None
                background.paste(img.convert("RGB"), mask=alpha)
                img = background
            else:
                img = img.copy()
            max_side = min(_VLM_MAX_IMAGE_SIDE, max(img.size))
            for quality in (85, 75, 65, 55):
                candidate = img.copy()
                candidate.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                out = io.BytesIO()
                candidate.save(out, format="JPEG", quality=quality, optimize=True)
                data = out.getvalue()
                if len(data) <= _VLM_MAX_PROVIDER_BYTES:
                    return data, "image/jpeg"
                max_side = max(64, int(max_side * 0.75))
    except Exception:
        log.debug("Failed to downscale VLM image payload", exc_info=True)
    if len(raw) <= _VLM_MAX_PROVIDER_BYTES:
        return raw, mime
    raise ValueError(
        f"⚠️ VLM_IMAGE_TOO_LARGE: image payload exceeds {int(_VLM_MAX_PROVIDER_BYTES / 1024 / 1024)}MB provider cap"
    )


def _image_payload_from_bytes(raw: bytes, mime: str) -> Dict[str, str]:
    import base64

    capped_raw, capped_mime = _downscale_image_for_vlm(raw, mime)
    return {"base64": base64.b64encode(capped_raw).decode(), "mime": capped_mime}


def _image_payload_from_base64(image_base64: str, mime: str) -> Dict[str, str]:
    import base64

    try:
        raw = base64.b64decode(image_base64, validate=True)
    except Exception:
        return {"base64": image_base64, "mime": mime}
    return _image_payload_from_bytes(raw, mime)


_VLM_NO_VISION_MODEL_MSG = (
    "⚠️ VLM_NO_VISION_MODEL: image analysis is unavailable — neither the active "
    "model nor any configured vision slot (light/heavy/main/fallback) accepts image "
    "input. Do NOT retry the image. Instead inspect the page as TEXT/DOM "
    "(browse_page output='html' or 'text') and the console/network for errors, or "
    "switch_model to a vision-capable model, or ask the owner to configure one."
)


def _vision_capable_slot_candidates(client: Any, ctx: Any = None) -> List[str]:
    """Configured models that may serve a VLM sub-call, most-local/cheapest first
    (active task model -> light -> heavy -> main -> fallback chain). Reviewer/scope slots
    are deliberately NOT poached. De-duplicated, order-preserving, empties dropped."""
    out: List[str] = [
        str(getattr(ctx, "active_model", "") or getattr(ctx, "task_model_override", "") or "").strip(),
    ]
    try:
        # Resolve the light + heavy slots through their configured accessors (P7), which
        # fall back to Main when the slot is empty (the v6.39 role-model default), instead
        # of a bare env read that would yield nothing for an unset slot.
        from ouroboros.config import get_heavy_model, get_light_model, get_vision_model
        out.append(str(get_vision_model() or "").strip())
        out.append(str(get_light_model() or "").strip())
        out.append(str(get_heavy_model() or "").strip())
    except Exception:
        out.append(str(os.environ.get("OUROBOROS_MODEL_HEAVY", "") or "").strip())
    try:
        out.append(str(client.default_model() or "").strip())
    except Exception:
        pass
    out.append(str(os.environ.get("OUROBOROS_MODEL", "") or "").strip())
    # Fallbacks is a comma chain -> add each link as its own candidate (via the shared
    # SSOT parser, which also honors the legacy singular env), not the raw comma-string
    # (which would never match a vision-capable model id).
    try:
        from ouroboros.config import parse_fallback_chain
        out.extend(parse_fallback_chain())
    except Exception:
        pass
    seen: set = set()
    uniq: List[str] = []
    for model in out:
        if model and model not in seen:
            seen.add(model)
            uniq.append(model)
    return uniq


def _resolve_vlm_model(client: Any, requested_model: str = "", *, ctx: Any = None) -> str:
    """Resolve a VISION-CAPABLE model for an image sub-call, or "" when none is
    available. An explicit requested model is honored ONLY if it actually supports
    vision (else "" -> the caller surfaces a typed capability gap, never a blind 404
    that the loop then bangs on). Otherwise route to the first vision-capable
    configured slot (active -> light -> heavy -> main -> fallback) — a gemini light/main
    is vision-capable, so this usually succeeds without any new model slot."""
    from ouroboros.provider_models import supports_vision
    requested = str(requested_model or "").strip()
    if requested:
        return requested if supports_vision(requested) else ""
    for candidate in _vision_capable_slot_candidates(client, ctx):
        if supports_vision(candidate):
            return candidate
    return ""


def _allowed_file_roots(ctx: Any = None) -> List["pathlib.Path"]:
    """Roots a VLM file_path may be read from: the uploads dir PLUS — same trust
    boundary the agent already has via read_file/run_command — the ACTIVE task
    workspace, so it can analyze a screenshot it just produced. Never arbitrary
    filesystem paths (no exfiltration surface the agent doesn't already hold)."""
    import pathlib
    data_dir = os.environ.get("OUROBOROS_DATA_DIR", "")
    if data_dir:
        _base = pathlib.Path(data_dir).expanduser().resolve()
    else:
        _base = pathlib.Path("~/Ouroboros/data").expanduser().resolve()
    # uploads PLUS skill job/state outputs (state/skills/<name>/jobs/...): trusted
    # local files the agent's OWN reviewed skills produce (e.g. computer-use
    # screenshots). Same trust boundary as read_file; view_image is image-only with
    # a fail-closed MIME sniff + size cap, so this adds no new exfiltration surface.
    roots = [_base / "uploads", _base / "state" / "skills"]
    if ctx is not None:
        try:
            from ouroboros.tools.registry import active_repo_dir_for
            roots.append(pathlib.Path(active_repo_dir_for(ctx)).expanduser().resolve())
        except Exception:
            pass
        # C3: the active task's first-class artifact roots (artifact_store +
        # task_drive) are the SAME trust boundary the agent already holds via
        # read_file/run_command — so a screenshot it just registered as an artifact
        # is readable too. Never arbitrary paths (no new exfiltration surface).
        for _root in ("artifact_store", "task_drive"):
            try:
                from ouroboros.tool_access import resource_root_path
                roots.append(pathlib.Path(resource_root_path(ctx, _root)).expanduser().resolve())
            except Exception:
                pass
    return roots


def _load_local_image_payload(ctx: ToolContext, file_path: str) -> Tuple[Optional[Dict[str, str]], str]:
    """Validate a LOCAL image path against the SAME trust boundary the agent already
    holds via read_file/run_command (allowed roots + protected-artifact read_bytes
    policy + size cap + fail-closed MIME sniff), then return a downscaled provider
    payload ``{"base64", "mime"}``. On any rejection returns ``(None, message)``.
    LOCAL FILES ONLY — no URL, no base64 (no new exfiltration surface). Shared by
    vlm_query(file_path=...) and view_image so both enforce identical checks."""
    import pathlib
    fp = pathlib.Path(file_path).expanduser().resolve()
    if not fp.exists():
        return None, f"⚠️ File not found: {file_path}"
    allowed = _allowed_file_roots(ctx)
    if not any(_path_is_under(fp, root) for root in allowed):
        return None, (
            f"⚠️ file_path must be inside the uploads directory, the skill-state tree "
            f"(state/skills), the active task workspace, or the task's "
            f"artifact_store/task_drive. Resolved path: {fp}. Use read_file for other paths."
        )
    # Honor the task protected-artifact policy: a workspace file may still be a
    # black-box protected artifact whose bytes must not be read (same contract as
    # read_file / query_code — block_reason_for_path with operation "read_bytes").
    try:
        from ouroboros.protected_artifacts import block_reason_for_path
        _artifact_block = block_reason_for_path(ctx, fp, "read_bytes")
    except Exception:
        _artifact_block = ""
    if _artifact_block:
        return None, _artifact_block
    if fp.stat().st_size > _VLM_MAX_FILE_BYTES:
        return None, f"⚠️ File too large ({fp.stat().st_size} bytes). Max {_VLM_MAX_FILE_BYTES} bytes."
    try:
        raw = fp.read_bytes()
    except Exception as e:
        return None, f"⚠️ Failed to read image file: {e}"
    # Fail closed: only recognized image bytes may be used.
    mime = _detect_image_mime_for_vlm(raw)
    if not mime:
        return None, (
            "⚠️ File does not appear to be a supported image (PNG/JPEG/GIF/WEBP). "
            "Only image files are accepted."
        )
    try:
        return _image_payload_from_bytes(raw, mime), ""
    except ValueError as e:
        return None, str(e)


def _vlm_query(ctx: ToolContext, prompt: str, image_url: str = "", image_base64: str = "", image_mime: str = "image/png", file_path: str = "", model: str = "") -> str:
    """Analyze one image from uploads file_path, public URL, or base64."""
    if not image_url and not image_base64 and not file_path:
        return "⚠️ Provide one of: file_path, image_url, or image_base64."

    images: List[Dict[str, Any]] = []
    try:
        if file_path:
            payload, err = _load_local_image_payload(ctx, file_path)
            if err:
                return err
            images.append(payload)
        elif image_url:
            images.append({"url": image_url})
        else:
            images.append(_image_payload_from_base64(image_base64, image_mime))

        client = _get_llm_client()
        vlm_model = _resolve_vlm_model(client, model, ctx=ctx)
        if not vlm_model:
            return _VLM_NO_VISION_MODEL_MSG
        text, usage = _vision_query_with_timeout(
            client,
            prompt=prompt,
            images=images,
            model=vlm_model,
            reasoning_effort=resolve_effort("task"),
            timeout=_VLM_HTTP_TIMEOUT_SEC,
        )

        _emit_usage(ctx, usage, vlm_model)

        return text or "(no response from VLM)"
    except Exception as e:
        log.warning("vlm_query failed: %s", e, exc_info=True)
        return f"⚠️ VLM_QUERY_FAILED: {e}"


def _emit_usage(ctx: ToolContext, usage: Dict[str, Any], model: str) -> None:
    """Emit LLM usage event for budget tracking."""
    if ctx.event_queue is None:
        return
    try:
        event = {
            "type": "llm_usage",
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
            "cost": usage.get("cost", 0.0),
            "task_id": ctx.task_id,
            "task_type": ctx.current_task_type or "task",
        }
        ctx.event_queue.put_nowait(event)
    except Exception:
        log.debug("Failed to emit VLM usage event", exc_info=True)


def _view_image(ctx: ToolContext, path: str = "") -> str:
    """Bring a LOCAL image file into the active model's context NATIVELY.

    Resource class: local_file_to_model (NOT a web tool — it never touches the
    network, so it is available even under allowed_resources.web=false). For a
    vision-capable active remote route the image is injected as a native image
    content block (the agent sees it INLINE in its own reasoning, like a browser
    screenshot); send-time routing may caption/omit for blind/local routes. LOCAL PATHS ONLY
    (no URL / no base64), same trust boundary as read_file. Prefer this over
    vlm_query when you need to reason about the image yourself (charts, renders,
    screenshots, photos, scanned/printed text)."""
    if not path:
        return "⚠️ Provide a local image file path."
    payload, err = _load_local_image_payload(ctx, path)
    if err:
        return err
    b64, mime = payload["base64"], payload["mime"]

    messages = getattr(ctx, "messages", None)
    if not isinstance(messages, list):
        return "⚠️ VIEW_IMAGE_UNAVAILABLE: no active conversation to attach the image to."

    import pathlib
    import base64 as _b64
    from ouroboros.utils import utc_now_iso

    src_name = pathlib.Path(path).name
    ts = utc_now_iso().replace(":", "").replace("-", "")[:15]
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}.get(mime, "img")
    view_dir = pathlib.Path(ctx.drive_root) / "uploads" / "views"
    try:
        view_dir.mkdir(parents=True, exist_ok=True)
        # Use the stem + the ACTUAL (possibly downscaled, e.g. PNG->JPEG) mime extension —
        # src_name already carries an extension, so f"{src_name}.{ext}" would double it.
        view_path = view_dir / f"{ts}_{pathlib.Path(path).stem}.{ext}"
        view_path.write_bytes(_b64.b64decode(b64))
        source_path = str(view_path)
    except Exception:
        source_path = str(pathlib.Path(path).expanduser().resolve())

    caption = f"[image: {src_name}]"
    from ouroboros.loop import _append_or_merge_user_content

    _append_or_merge_user_content(messages, [
        {"type": "text", "text": caption},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
            "_caption": caption,
            "_source_path": source_path,
        },
    ])
    return (
        f"'{src_name}' is now attached as a local image block. Vision-capable remote routes can "
        f"inspect it inline; blind/local routes may receive a caption or placeholder at send time. "
        f"It was read from local disk; this is NOT a web tool."
    )


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="analyze_screenshot",
            schema={
                "name": "analyze_screenshot",
                "description": (
                    "Analyze the last browser screenshot using a Vision LLM. "
                    "Must call browse_page(output='screenshot') or browser_action(action='screenshot') first. "
                    "Returns a text description and analysis of the screenshot. "
                    "Use this to verify UI, check for visual errors, or understand page layout. "
                    "For MEDIA CONTENT (a video/image inside the page), prefer extract_video_frames + "
                    "view_image on the source file over screenshotting a compressed player rendering — "
                    "a clean frame beats a low-res player capture."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to look for or analyze in the screenshot (default: general description)",
                        },
                        "model": {
                            "type": "string",
                            "description": "VLM model to use. Empty uses the active/vision slot resolution (OUROBOROS_MODEL_VISION empty->Main, then light/heavy/main/fallback candidates).",
                        },
                    },
                    "required": [],
                },
            },
            handler=_analyze_screenshot,
            timeout_sec=90,
        ),
        ToolEntry(
            name="vlm_query",
            schema={
                "name": "vlm_query",
                "description": (
                    "Analyze any image using a Vision LLM. "
                    "Provide one of: file_path (local file, preferred — avoids large base64 in arguments), "
                    "image_url (public URL), or image_base64 (base64-encoded PNG/JPEG). "
                    "Use file_path for files already on disk (e.g. data/uploads/ attachments). "
                    "Use for: analyzing charts, reading diagrams, understanding screenshots, checking UI. "
                    "NOTE: this DELEGATES to a separate vision model — when you are vision-capable "
                    "yourself, prefer view_image (native inline vision, no second-model handoff) for "
                    "anything you need to REASON about rather than merely describe."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to analyze or describe about the image",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Local file path to image (preferred — reads from disk, avoids base64 in arguments). Must be inside the uploads directory (data/uploads/), the skill-state tree (data/state/skills, e.g. a computer-use screenshot), the active task workspace, or the task's artifact_store/task_drive (e.g. artifact_store/video_frames frames, artifact_store/attachments staged files).",
                        },
                        "image_url": {
                            "type": "string",
                            "description": "Public URL of the image to analyze",
                        },
                        "image_base64": {
                            "type": "string",
                            "description": "Base64-encoded image data",
                        },
                        "image_mime": {
                            "type": "string",
                            "description": "MIME type for base64 image (default: image/png)",
                        },
                        "model": {
                            "type": "string",
                            "description": "VLM model to use. Empty uses the active/vision slot resolution (OUROBOROS_MODEL_VISION empty->Main, then light/heavy/main/fallback candidates).",
                        },
                    },
                    "required": ["prompt"],
                },
            },
            handler=_vlm_query,
            timeout_sec=90,
        ),
        ToolEntry(
            name="view_image",
            schema={
                "name": "view_image",
                "description": (
                    "Bring a LOCAL image file natively into your own context so you can SEE and reason "
                    "about it directly (vision-capable models). Resource class: local_file_to_model — it "
                    "reads a local file and attaches it into your context; it is NOT a web tool and works "
                    "even when web/network access is disabled. LOCAL PATHS ONLY (inside the task workspace, "
                    "uploads dir, or the task's artifact_store/task_drive — e.g. frames from "
                    "extract_video_frames under artifact_store/video_frames, or staged attachments under "
                    "artifact_store/attachments); no URLs. Typical flow: after list_files reveals an image file "
                    "(.png/.jpg/.jpeg/.gif/.webp) — including one you rendered yourself, e.g. a chart or a "
                    "rendered toolpath — call view_image(path) and then analyze it inline. Prefer this over "
                    "vlm_query when you need to reason about the image yourself rather than ask a separate model."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Local image file path inside the task workspace, uploads dir, the skill-state tree (data/state/skills), or the task's artifact_store/task_drive (e.g. /app/chart.png after list_files finds it, or artifact_store/video_frames/frame_001.png from extract_video_frames).",
                        },
                    },
                    "required": ["path"],
                },
            },
            handler=_view_image,
            timeout_sec=30,
        ),
    ]
